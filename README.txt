LDTTS is a Python program that captures microphone audio, transcribes it in real time (usually <1s
latency), and repeats it with pyttsx3 (or Klattsch, more on that further down) routed to a virtual
audio cable. it's great for streaming, roleplay, or just dicking around with a robotic voice.

make sure you have deepgram_api_key set in config.json or this won't work. as of writing this
(6/30/2026) Deepgram gives you $200 of free credit without needing to sign up for some free
trial when you make an account, so this shouldn't be too much of a hassle to set up. the reason
i'm using Deepgram rather than a typical speech-to-text library is because of its 'per-word'
functionality which works much better for this program than anything else i've been able to find.

make sure you actually set your own input and output devices in config.json (input_device and
output_device respectively). your input device should be the microphone that you use, and your
output device should be a virtual cable output that you can then use as your input device for
Discord or games or whatever. i'd highly recommend using VoiceMeeter.
check https://vb-audio.com/Cable/ for a free audio cable. this is already the default output
device in config.json.

to use Klattsch, go to https://github.com/tgies/klattsch, download the source code as zip,
unzip it, and put the contents of the innermost 'klattsch-main' folder into guts/utilities
(its internal files should be siblings to klattsch.py in the directory, so you should see
like index.html next to it). i just don't want to put Klattsch in this myself because reasons
(klattsch.py is just a Python wrapper for Klattsch, you can tear it out and use it yourself,
but it probably needs work). in my experience, while i love it, Klattsch is pretty slow, so you
might just want to stick to pyttsx3. older iterations of this project had in-house concatenative
voicebank text to speech capabilities, but it was too jank. (feel free to add this yourself if
you're smarter than me)

to change between pyttsx3 and klattsch (or any other voice engines in the future), literally
change tts_engine in the config to "klattsch" or "pyttsx3"

note that this program does stream audio from your microphone to Deepgram and i can't personally
verify their security, so use this at your own risk

also!!! about 90% of this is shitty ai spaghetti code and i've made some potentially questionable
design decisions (though it does work pretty well in my experience), so three things:
    1. issues and weird redundancy are likely
    2. i don't really care all that much about being credited
    3. forks are appreciated

once this is set up, just run ldtts.py to use

P.S. i would highly recommend not touching the config beyond input_device, output_device,
deepgram_api_key, and tts_engine, but if you're messing with timings, audio_overlap_buffer is how
much generated audio files overlap when being played out of the queue. if you're having issues or
you're working on a derivative or whatever, you can find me at https://discord.gg/YSsJhPmD2m or
@livingdyad on Discord. obviously, you don't *have* to notify me if you're working on
derivative of this, but i would like to see... :eyes:
