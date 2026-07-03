import asyncio
import json
import os
import threading
import argparse
import functools
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional, Dict, Any, Tuple
from multiprocessing import Queue, Process
import numpy as np
import sounddevice as sd
from collections import deque
from deepgram import DeepgramClient
from deepgram.core.events import EventType

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - optional dependency
    AudioSegment = None

try:
    from g2p_en import G2p
except ImportError:  # pragma: no cover - optional dependency
    G2p = None

# ----------------- Config -----------------
DEFAULTS = {
    "deepgram_api_key": None,
    "deepgram_model": "flux-general-en",
    "deepgram_language": "en-US",
    "deepgram_encoding": "linear16",
    "deepgram_punctuate": True,
    "deepgram_interim_results": False,
    "sample_rate": 16000,
    "channels": 1,
    "input_device": None,
    "device_hint": None,
    "device_index": None,
    "frame_duration": 0.18,
    "debug": True,
    "audio_overlap_buffer": 0.1,
    "output_device": None,
    "tts_engine": "samtts",
    "tts_rate": 150,
    "pyttsx3_append_period": False,
    "audio_trim_silence_thresh": -45,
    "audio_trim_silence_len": 25,
    "audio_trim_padding": 5,
}


def write_status_line(message: str) -> None:
    sys.stdout.write(f"\r\x1b[2K{message}")
    sys.stdout.flush()


class StartupSpinner:
    def __init__(self, message: str, enabled: bool = True):
        self.message = message
        self.enabled = enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            write_status_line(self.message)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, message: str) -> None:
        self.message = message

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        write_status_line("")

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        idx = 0
        while not self._stop_event.is_set():
            write_status_line(f"{self.message} {frames[idx % len(frames)]}")
            idx += 1
            time.sleep(0.1)


def load_config(path: str) -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                cfg.update(json.load(f))
    except Exception as e:
        print(f"Warning: failed to read config {path}: {e}")

    if not cfg.get("deepgram_api_key"):
        cfg["deepgram_api_key"] = os.environ.get("DEEPGRAM_API_KEY")

    cfg["deepgram_model"] = str(cfg.get("deepgram_model", DEFAULTS["deepgram_model"]))
    cfg["deepgram_language"] = str(cfg.get("deepgram_language", DEFAULTS["deepgram_language"]))
    cfg["deepgram_encoding"] = str(cfg.get("deepgram_encoding", DEFAULTS["deepgram_encoding"]))
    cfg["deepgram_punctuate"] = bool(cfg.get("deepgram_punctuate", DEFAULTS["deepgram_punctuate"]))
    cfg["deepgram_interim_results"] = bool(cfg.get("deepgram_interim_results", DEFAULTS["deepgram_interim_results"]))
    cfg["sample_rate"] = int(cfg.get("sample_rate", DEFAULTS["sample_rate"]))
    cfg["channels"] = int(cfg.get("channels", DEFAULTS["channels"]))
    if cfg.get("device_index") is not None:
        cfg["device_index"] = int(cfg["device_index"])
    cfg["debug"] = bool(cfg.get("debug", DEFAULTS["debug"]))
    cfg["frame_duration"] = float(cfg.get("frame_duration", DEFAULTS["frame_duration"]))
    cfg["audio_overlap_buffer"] = float(cfg.get("audio_overlap_buffer", DEFAULTS["audio_overlap_buffer"]))
    cfg["audio_trim_silence_thresh"] = float(cfg.get("audio_trim_silence_thresh", DEFAULTS["audio_trim_silence_thresh"]))
    cfg["audio_trim_silence_len"] = int(cfg.get("audio_trim_silence_len", DEFAULTS["audio_trim_silence_len"]))
    cfg["audio_trim_padding"] = int(cfg.get("audio_trim_padding", DEFAULTS["audio_trim_padding"]))
    cfg["tts_rate"] = max(1, int(cfg.get("tts_rate", DEFAULTS["tts_rate"])))
    cfg["pyttsx3_append_period"] = bool(cfg.get("pyttsx3_append_period", DEFAULTS["pyttsx3_append_period"]))
    return cfg

def load_word_replacements(path: Optional[str] = None) -> Dict[str, str]:
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordreplacements.json")
    replacements_path = path or default_path
    try:
        if os.path.exists(replacements_path):
            with open(replacements_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception as exc:
        print(f"Warning: failed to load word replacements from {replacements_path}: {exc}")
    return {}


WORD_REPLACEMENTS = load_word_replacements()


def normalize_word_for_tts(word: str) -> str:
    w = word.lower()

    # Direct replacement
    if w in WORD_REPLACEMENTS:
        return WORD_REPLACEMENTS[w]

    return word


def normalize_text_for_tts(text: str) -> str:
    if not text:
        return ""

    def replace_match(match: re.Match[str]) -> str:
        return normalize_word_for_tts(match.group(0))

    return re.sub(r"\b[\w']+\b", replace_match, str(text))


def prepare_text_for_tts(text: str, full_text_mode: bool = False) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    if full_text_mode:
        return cleaned
    return normalize_word_for_tts(cleaned) + "."


def trim_g2p_stress_digits(token: str) -> str:
    return re.sub(r"\d+", "", token).strip()


def sanitize_klattsch_text(text: str) -> str:
    cleaned_text = " ".join(str(text).split())
    if not cleaned_text:
        return ""
    cleaned_text = re.sub(r"[^\x00-\x7F]", " ", cleaned_text)
    cleaned_text = re.sub(r"[\.,;:!?()\[\]{}<>\"“”‘’*+#=~`@_$%^&|\\]", " ", cleaned_text)
    cleaned_text = cleaned_text.replace("'", "'")
    return re.sub(r"\s+", " ", cleaned_text).strip()

def apply_prosody_to_phonemes(text: str) -> str:
    if not text:
        return ""

    tokens = text.split()
    out = []

    for tok in tokens:
        if not tok:
            continue

        comma = tok.endswith(",")
        period = tok.endswith(".")
        question = tok.endswith("?")
        exclaim = tok.endswith("!")

        base = tok.rstrip(",.?!")

        token = base

        # --- PROSODY (POST-PHONEME SAFE) ---

        if question:
            # rising contour (safe postfix ONLY)
            token = f"{token}(+15)"

        elif exclaim:
            # emphasis: pitch + stress marker
            token = f"bC5 {token}"

        elif comma:
            token = f"{token};"

        elif period:
            token = f"{token};"

        out.append(token)

    # normalize spacing of pauses
    cleaned = " ".join(out)
    cleaned = cleaned.replace(" ;", ";")

    return cleaned

def text_to_klattsch_phonemes(text: str) -> str:
    cleaned_text = sanitize_klattsch_text(text)
    if not cleaned_text:
        return ""

    # 🔍 DEBUG: what is ACTUALLY entering G2P
    print("\n" + "=" * 60)
    print("[G2P INPUT RAW]:", repr(text))
    print("[G2P INPUT CLEANED]:", repr(cleaned_text))
    print("=" * 60 + "\n")

    if G2p is None:
        raise RuntimeError("g2p-en is required for klattsch TTS. Install it with 'pip install g2p-en'.")

    try:
        g2p = G2p()
        phonemes = g2p(cleaned_text)

        # 🔍 DEBUG: what G2P outputs
        print("[G2P OUTPUT]:", phonemes)

    except Exception as exc:
        if "CMUDictCorpusReader" in str(exc) or "LazyCorpusLoader" in str(exc):
            return cleaned_text
        raise RuntimeError(f"Failed to convert text to phonemes with g2p-en: {exc}") from exc

    cleaned_tokens = []
    for phoneme in phonemes:
        if not isinstance(phoneme, str):
            continue
        cleaned = trim_g2p_stress_digits(phoneme)
        if cleaned:
            cleaned_tokens.append(cleaned)

    final = " ".join(cleaned_tokens)

    # 🔍 DEBUG: final phoneme string
    print("[G2P FINAL STRING]:", final)
    print("=" * 60 + "\n")

    return final


# ----------------- Audio -----------------
def find_input_device_by_name(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and str(name).strip().lower() == str(d.get("name", "")).strip().lower():
            return i
    return None


def find_output_devices() -> list[int]:
    """Find all available output devices"""
    try:
        devices = sd.query_devices()
        output_devices = []
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0:
                output_devices.append(i)
        return output_devices
    except Exception:
        return []

def find_output_device_by_name(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and str(name).strip().lower() == str(d.get("name", "")).strip().lower():
                return i
    except Exception:
        pass
    return None


def list_input_devices() -> list[str]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    return [str(d.get("name", "")) for d in devices if d.get("max_input_channels", 0) > 0]


def list_output_devices() -> list[str]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    return [str(d.get("name", "")) for d in devices if d.get("max_output_channels", 0) > 0]

# Runtime queues
audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=50)
word_queue: "asyncio.Queue[tuple]" = asyncio.Queue()
ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
spoken_word_starts = set()
spoken_words = set()
printing_started = False
has_printed_words = False
pending_words: Dict[int, str] = {}  # Maps start_index to word text
last_dequeued_index: int = -1  # Track the highest start_index we've already added to queue
queue_position: int = 0  # Track position of words in the output queue
audio_player_queues = None  # Will hold [Queue, Queue] for 2 audio players
mixed_playback_queue = None  # Will hold Queue for mixed outputs
audio_player_processes = None  # Will hold [Process, Process]
current_player_index = 0  # Track which subprocess to send next audio to

def audio_callback(sample_rate, channels, debug, indata, frames, time_info, status):
    try:
        data = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        pcm16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16) if data.dtype != np.int16 else data.astype(np.int16)
        audio_bytes = pcm16.tobytes()
        global ASYNC_LOOP
        if ASYNC_LOOP and ASYNC_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(audio_queue.put(audio_bytes), ASYNC_LOOP)
    except Exception as e:
        if debug:
            print("[audio_callback] ERROR:", repr(e))

async def audio_sender(connection, sample_rate: int, channels: int, debug: bool = False):
    while True:
        try:
            audio_data = await audio_queue.get()
            connection.send_media(audio_data)
        except asyncio.CancelledError:
            break
        except Exception as e:
            if debug:
                print("[audio_sender] ERROR:", repr(e))

# ----------------- Audio Player Subprocess Worker -----------------
def subprocess_audio_player_worker(queue, mixed_queue):
    """Worker process for generating and emitting per-track streams for mixing"""
    while True:
        try:
            item = queue.get()
            if item is None:  # Sentinel to stop process
                break

            audio_data, sample_rate = item
            try:
                # Send generated track to mixer queue
                mixed_queue.put((audio_data, sample_rate))
            except Exception as e:
                print(f"[subprocess_audio_player_worker] ERROR: {e}")
        except Exception as e:
            print(f"[subprocess_audio_player_worker] Queue ERROR: {e}")
            break

# ----------------- Mixer Thread -----------------
def mixer_thread_fn(mixed_queue, sample_rate, channels, output_device):
    active_streams = []  # list of (samples, position)
    lock = threading.Lock()

    def callback(outdata, frames, time_info, status):
        nonlocal active_streams
        with lock:
            outbuf = np.zeros((frames, channels), dtype=np.float32)
            # Pull new streams from mixed_queue non-blocking
            while True:
                try:
                    samples, sr = mixed_queue.get_nowait()
                    # Expect sr == sample_rate
                    if samples.ndim == 1 and channels == 2:
                        samples = np.repeat(samples[:, np.newaxis], 2, axis=1)
                    elif samples.ndim == 1:
                        samples = samples.reshape((-1, 1))
                    active_streams.append([samples, 0])
                except Exception:
                    break

            if not active_streams:
                outdata.fill(0)
                return

            remaining_streams = []
            for samples, pos in active_streams:
                end = min(pos + frames, samples.shape[0])
                chunk = samples[pos:end]
                if chunk.shape[0] < frames:
                    chunk = np.pad(chunk, ((0, frames - chunk.shape[0]), (0, 0)), mode='constant')
                outbuf[:frames] += chunk
                if end < samples.shape[0]:
                    remaining_streams.append([samples, end])

            gain = 0.6  # adjust between 0.3–0.8 as needed
            outbuf *= 0.7
            outbuf = np.tanh(outbuf)
            outdata[:] = outbuf
            active_streams = remaining_streams

    with sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype='float32',
        callback=callback,
        blocksize=1024,
        device=output_device
    ):
        while True:
            time.sleep(0.01)

# ----------------- TTS Functions -----------------
def trim_audio_for_playback(audio: Any, cfg: Optional[Dict[str, Any]] = None) -> Any:
    if audio is None or len(audio) == 0:
        return audio

    cfg = cfg or DEFAULTS
    silence_thresh = float(cfg.get("audio_trim_silence_thresh", DEFAULTS["audio_trim_silence_thresh"]))
    silence_len = int(cfg.get("audio_trim_silence_len", DEFAULTS["audio_trim_silence_len"]))
    padding = int(cfg.get("audio_trim_padding", DEFAULTS["audio_trim_padding"]))

    try:
        silence_len = max(10, min(200, silence_len))
        padding = max(0, min(50, padding))
        return audio.strip_silence(silence_thresh=silence_thresh, silence_len=silence_len, padding=padding)
    except Exception:
        return audio


async def generate_tts_audio_indexed(word: str, index: int, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Generate TTS audio and queue for subprocess playback, returns duration"""
    try:
        loop = asyncio.get_event_loop()
        duration = await loop.run_in_executor(None, _generate_and_queue_tts, word, cfg)
        return duration if duration else 1.0
    except Exception as e:
        print(f"[generate_tts_audio_indexed] ERROR: {e}")
        return 1.0


def _generate_and_queue_tts(text: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Generate TTS and queue for subprocess playback, returns duration"""
    global current_player_index, audio_player_queues
    cfg = cfg or DEFAULTS
    sample_rate = int(cfg.get("sample_rate", DEFAULTS["sample_rate"]))
    preferred_engine = str(cfg.get("tts_engine", DEFAULTS.get("tts_engine", "samtts"))).lower()

    try:
        # Generate to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            temp_path = f.name

        if preferred_engine == "klattsch":
            wrapper_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "guts",
                "utilities",
                "klattsch-main",
                "klattsch.py",
            )
            phoneme_text = text
            if not phoneme_text:
                raise RuntimeError("No phonemes available for klattsch synthesis")
            result = subprocess.run(
                [sys.executable, wrapper_path, phoneme_text, "--output", temp_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "klattsch failed with exit code {}\nstdout:\n{}\nstderr:\n{}".format(
                        result.returncode,
                        result.stdout.strip(),
                        result.stderr.strip(),
                    )
                )
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
        else:
            if pyttsx3 is None:
                raise RuntimeError("pyttsx3 is required for the default TTS engine")
            if AudioSegment is None:
                raise RuntimeError("pydub is required for the default TTS engine")

            engine = pyttsx3.init()
            tts_rate = int(cfg.get("tts_rate", DEFAULTS.get("tts_rate", 150)))
            engine.setProperty('rate', max(1, tts_rate))
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

        # Load audio
        try:
            if AudioSegment is None:
                raise RuntimeError("pydub is required to read generated audio")

            audio = AudioSegment.from_wav(temp_path)
            # Trim silence from beginning and end without tripping on short clips
            audio = trim_audio_for_playback(audio, cfg)
            # Force target rate to match the stream samplerate
            audio = audio.set_frame_rate(sample_rate)
            duration = len(audio) / 1000.0  # Convert ms to seconds

            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # Queue for subprocess playback (round-robin between 2)
            if audio_player_queues:
                sub_idx = current_player_index
                out_queue = audio_player_queues[sub_idx]
                current_player_index = 1 - current_player_index

                # Normalize to float32 for mixing
                samples_f32 = (samples.astype(np.float32) / 32768.0)
                out_queue.put((samples_f32, int(sample_rate)))

            return duration
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"[_generate_and_queue_tts] ERROR: {e}")
        return 1.0

# ----------------- Word Queue Printer -----------------
async def print_words_from_queue(cfg: Dict[str, Any]):
    global printing_started, has_printed_words

    audio_overlap_buffer = cfg.get("audio_overlap_buffer", 0.1)
    preferred_engine = str(cfg.get("tts_engine", DEFAULTS.get("tts_engine", "samtts"))).lower()
    use_klattsch = preferred_engine == "klattsch"

    try:
        while True:
            queue_idx, text = await word_queue.get()
            text = normalize_text_for_tts(text)

            if not printing_started:
                printing_started = True
                has_printed_words = True

            if not text:
                continue

            if cfg.get("debug", False):
                print("\n============================================================")
                print(f"[TTS INPUT NORMALIZED]: {text}")
                print("============================================================")

            if use_klattsch:
                # STEP 1: CLEAN TEXT → PHONEMES
                phonemes = text_to_klattsch_phonemes(text)

                if cfg.get("debug", False):
                    print(f"[G2P OUTPUT]: {phonemes}")
                    print("============================================================")

                if not phonemes:
                    continue

                # STEP 2: APPLY PROSODY (ONLY HERE)
                final = apply_prosody_to_phonemes(phonemes)

                if cfg.get("debug", False):
                    print(f"[FINAL PHONEME STRING]: {final}")
                    print("============================================================")

                tts_input = final
            else:
                tts_input = text
                if cfg.get("pyttsx3_append_period", False):
                    tts_input = tts_input.rstrip() + "."

            write_status_line(f"this muffucka on some \"{text}\"")

            duration = await generate_tts_audio_indexed(tts_input, queue_idx, cfg)

            await asyncio.sleep(max(0.0, duration - audio_overlap_buffer))

            while True:
                try:
                    queue_idx, text = word_queue.get_nowait()
                    text = normalize_text_for_tts(text)

                    if cfg.get("debug", False):
                        print("\n============================================================")
                        print(f"[TTS INPUT NORMALIZED]: {text}")
                        print("============================================================")

                    if use_klattsch:
                        phonemes = text_to_klattsch_phonemes(text)

                        if cfg.get("debug", False):
                            print(f"[G2P OUTPUT]: {phonemes}")
                            print("============================================================")

                        if not phonemes:
                            continue

                        final = apply_prosody_to_phonemes(phonemes)
                        tts_input = final
                    else:
                        tts_input = text
                        if cfg.get("pyttsx3_append_period", False):
                            tts_input = tts_input.rstrip() + "."

                    write_status_line(f"this muffucka on some \"{text}\"")

                    duration = await generate_tts_audio_indexed(tts_input, queue_idx, cfg)

                    await asyncio.sleep(max(0.0, duration - audio_overlap_buffer))

                except asyncio.QueueEmpty:
                    printing_started = False
                    break

    except asyncio.CancelledError:
        pass

# ----------------- Recorder -----------------
def recorder_thread_fn(device: Optional[int], sample_rate: int, channels: int, frame_duration: float, debug: bool):
    blocksize = max(int(frame_duration * sample_rate), 2048)
    try:
        callback = functools.partial(audio_callback, sample_rate, channels, debug)
        with sd.InputStream(device=device, samplerate=sample_rate, channels=channels, dtype='float32',
                            blocksize=blocksize, callback=callback):
            while True:
                sd.sleep(1000)  # Keep the stream alive
    except Exception as e:
        if debug:
            print("[recorder_thread_fn] ERROR:", repr(e))

# ----------------- Main Async -----------------
async def run_realtime(cfg: Dict[str, Any]):
    global ASYNC_LOOP, audio_player_queues, mixed_playback_queue, audio_player_processes, current_player_index
    ASYNC_LOOP = asyncio.get_running_loop()
    progress_enabled = bool(cfg.get("debug", DEFAULTS["debug"])) or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
    spinner = StartupSpinner("[startup] Initializing audio pipeline", enabled=progress_enabled)
    spinner.start()

    DEEPGRAM_API_KEY = cfg.get("deepgram_api_key")
    DEEPGRAM_MODEL = str(cfg.get("deepgram_model", DEFAULTS["deepgram_model"]))
    DEEPGRAM_LANGUAGE = str(cfg.get("deepgram_language", DEFAULTS["deepgram_language"]))
    DEEPGRAM_ENCODING = str(cfg.get("deepgram_encoding", DEFAULTS["deepgram_encoding"]))
    DEEPGRAM_PUNCTUATE = bool(cfg.get("deepgram_punctuate", DEFAULTS["deepgram_punctuate"]))
    DEEPGRAM_INTERIM_RESULTS = bool(cfg.get("deepgram_interim_results", DEFAULTS["deepgram_interim_results"]))
    TARGET_SAMPLE_RATE = int(cfg.get("sample_rate", DEFAULTS["sample_rate"]))
    CHANNELS = int(cfg.get("channels", DEFAULTS["channels"]))
    DEBUG = bool(cfg.get("debug", DEFAULTS["debug"]))
    FRAME_DURATION = float(cfg.get("frame_duration", DEFAULTS["frame_duration"]))

    if not DEEPGRAM_API_KEY:
        raise ValueError("Deepgram API key is required. Set it in the config or DEEPGRAM_API_KEY environment variable.")

    requested_input_device = str(cfg.get("input_device") or cfg.get("device_hint") or "").strip()
    requested_output_device = str(cfg.get("output_device") or "").strip()
    input_device_index = find_input_device_by_name(requested_input_device) if requested_input_device else None
    output_device_index = find_output_device_by_name(requested_output_device) if requested_output_device else None

    if input_device_index is None and output_device_index is None:
        write_status_line("")
        print("hey!!! input device and output device are invalid!!!")
        print("Input devices:")
        for name in list_input_devices() or ["(none found)"]:
            print(f"  - {name}")
        print("Output devices:")
        for name in list_output_devices() or ["(none found)"]:
            print(f"  - {name}")
        raise SystemExit(1)
    if input_device_index is None:
        write_status_line("")
        print("hey!!! input device is invalid!!!")
        print("Input devices:")
        for name in list_input_devices() or ["(none found)"]:
            print(f"  - {name}")
        raise SystemExit(1)
    if output_device_index is None:
        write_status_line("")
        print("hey!!! output device is invalid!!!")
        print("Output devices:")
        for name in list_output_devices() or ["(none found)"]:
            print(f"  - {name}")
        raise SystemExit(1)

    device = input_device_index

    spinner.update("[startup] Preparing playback workers")

    # Initialize audio player subprocesses
    current_player_index = 0
    output_devices = find_output_devices()
    if len(output_devices) >= 2:
        player_devices = [output_devices[0], output_devices[1]]
    else:
        default_device = output_devices[0] if output_devices else None
        player_devices = [default_device, default_device]

    audio_player_queues = [Queue(), Queue()]
    mixed_playback_queue = Queue()
    audio_player_processes = [
        Process(target=subprocess_audio_player_worker, args=(audio_player_queues[0], mixed_playback_queue)),
        Process(target=subprocess_audio_player_worker, args=(audio_player_queues[1], mixed_playback_queue))
    ]
    for p in audio_player_processes:
        p.start()

    spinner.update("[startup] Opening audio output stream")
    if DEBUG:
        print(f"[audio] Using output device index: {output_device_index}")

    mixer_thread = threading.Thread(
        target=mixer_thread_fn,
        args=(mixed_playback_queue, TARGET_SAMPLE_RATE, CHANNELS, output_device_index),
        daemon=True
    )

    mixer_thread.start()
    
    spinner.update("[startup] Connecting to Deepgram")

    deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    ready = threading.Event()
    history = deque(maxlen=3)
    committed_index = 0
    emitted_words = []

    def extract_words(text: str):
        return re.findall(r"\b[\w']+\b", text.lower())

    def stable_prefix():
        if not history:
            return []
        # Delay first-turn commits until at least a second frame arrives.
        # This gives the initial word time to stabilize before being emitted.
        if len(history) == 1:
            return []
        base = history[0]
        for i in range(len(base)):
            for frame in history:
                if i >= len(frame) or frame[i] != base[i]:
                    return base[:i]
        return base

    async def _enqueue_word(word: str) -> None:
        global queue_position
        await word_queue.put((queue_position, word))
        queue_position += 1

    def on_message(result):
        nonlocal committed_index, emitted_words
        transcript = result.get("transcript")
        event = result.get("event")

        if event == "StartOfTurn":
            if DEBUG:
                print("\n--- StartOfTurn ---")
            history.clear()
            committed_index = 0
            emitted_words = []

        if transcript:
            words = extract_words(transcript)
            history.append(words)
            stable = stable_prefix()

            while committed_index < len(stable):
                w = stable[committed_index]
                if DEBUG:
                    print(w)
                asyncio.run_coroutine_threadsafe(_enqueue_word(w), ASYNC_LOOP)
                committed_index += 1

        if event == "EndOfTurn":
            if DEBUG:
                print("--- EndOfTurn ---\n")

    with deepgram_client.listen.v2.connect(
        model=DEEPGRAM_MODEL,
        encoding=DEEPGRAM_ENCODING,
        sample_rate=TARGET_SAMPLE_RATE,
        eot_threshold=0.85,
        eot_timeout_ms=1200,
    ) as connection:
        connection.on(EventType.OPEN, lambda _: ready.set())
        connection.on(EventType.MESSAGE, on_message)

        listener_thread = threading.Thread(target=connection.start_listening, daemon=True)
        listener_thread.start()
        if not ready.wait(timeout=10):
            raise RuntimeError("Deepgram connection failed to open")

        try:
            spinner.stop()
            write_status_line("[startup] Connected; waiting for speech")

            sender_task = asyncio.create_task(audio_sender(connection, TARGET_SAMPLE_RATE, CHANNELS, DEBUG))
            printer_task = asyncio.create_task(print_words_from_queue(cfg))

            recorder_thread = threading.Thread(target=recorder_thread_fn, args=(device, TARGET_SAMPLE_RATE, CHANNELS, FRAME_DURATION, DEBUG))
            recorder_thread.start()

            try:
                await asyncio.gather(sender_task, printer_task)
            except KeyboardInterrupt:
                if DEBUG:
                    print("Interrupted by user")
            finally:
                recorder_thread.join()
        finally:
            if hasattr(connection, "close"):
                try:
                    connection.close()
                except Exception:
                    pass
            if hasattr(connection, "stop_listening"):
                try:
                    connection.stop_listening()
                except Exception:
                    pass
    spinner.stop()
    # Cleanup: send sentinels to stop subprocesses
    if audio_player_queues:
        for q in audio_player_queues:
            q.put(None)
    if audio_player_processes:
        for p in audio_player_processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Simple audio streamer that prints words from Deepgram transcription")
    parser.add_argument("--config", "-c", default=None, help="Path to JSON config file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.json")  # Updated default to 'config.json'

    cfg = load_config(config_path)

    try:
        asyncio.run(run_realtime(cfg))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
