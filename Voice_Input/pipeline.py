#!/usr/bin/env python3
"""Orchestrate the voice input workflow from microphone recording to transcription."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Voice_Input.record_audio import record_audio
from Voice_Input.Speech_to_text import speech_to_text


def run_pipeline() -> tuple[Path, Path]:
    """Record audio, transcribe it, and return the saved file paths."""
    audio_path = record_audio()
    text_path = speech_to_text(audio_path)
    transcription = text_path.read_text(encoding="utf-8")

    print("Transcription:")
    print(transcription)
    print(f"Audio saved at: {audio_path}")
    print(f"Text saved at: {text_path}")

    return audio_path, text_path


if __name__ == "__main__":
    run_pipeline()
