#!/usr/bin/env python3
"""Python wrapper for the local klattsch package.

This script calls the bundled Node CLI at "bin/klattsch.mjs" and makes it easy to
generate WAV files from text or phoneme strings.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def get_node_executable() -> str:
    node = shutil.which("node") or shutil.which("node.exe")
    if node:
        return node
    raise FileNotFoundError(
        "Node.js executable not found in PATH. Install Node.js or add it to PATH."
    )


def get_klattsch_cli_path() -> Path:
    root = Path(__file__).resolve().parent
    cli = root / "bin" / "klattsch.mjs"
    if not cli.exists():
        raise FileNotFoundError(f"Could not find klattsch CLI at {cli}")
    return cli


def generate_audio(text: str, output_path: str | Path, node_executable: str | None = None) -> Path:
    """Generate a WAV file from a klattsch text string."""
    node = node_executable or get_node_executable()
    out_path = Path(output_path)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".wav")

    cli = get_klattsch_cli_path()
    cmd = [node, str(cli), text, str(out_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "klattsch failed with exit code {}\nstdout:\n{}\nstderr:\n{}".format(
                result.returncode, result.stdout.strip(), result.stderr.strip()
            )
        )

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a WAV file from a klattsch phoneme string."
    )
    parser.add_argument(
        "text",
        help="Phoneme string or klattsch text. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="klattsch.wav",
        help="Output WAV filename (default: klattsch.wav)",
    )
    parser.add_argument(
        "--node",
        help="Path to the Node.js executable to use.",
    )
    args = parser.parse_args(argv)

    text = args.text
    if text == "-":
        text = sys.stdin.read().strip()
        if not text:
            parser.error("No text provided on stdin")

    output_path = generate_audio(text, args.output, args.node)
    # print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
