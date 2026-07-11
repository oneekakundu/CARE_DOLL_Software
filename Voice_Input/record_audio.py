#!/usr/bin/env python3
"""Record microphone audio from the terminal and save it as a WAV file."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def ensure_data_directory(base_dir: Path) -> Path:
    """Create a data folder if it does not already exist."""
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def create_output_path(data_dir: Path) -> Path:
    """Create a unique WAV file name using the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return data_dir / f"recording_{timestamp}.wav"


def record_audio(output_path: Path, samplerate: int = 44100, channels: int = 1) -> None:
    """Record audio until the user presses Enter to stop."""
    frames: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames_count: int, time_info, status) -> None:
        """Store audio chunks as they are captured."""
        if status:
            print(f"Audio status: {status}")
        frames.append(indata.copy())

    print("Press Enter to start recording.")
    input()

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            callback=callback,
        ):
            print("Recording...")
            print("Press Enter to stop.")
            input()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Recording cancelled.")
        raise
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Microphone not available or permission denied.") from exc

    if not frames:
        raise RuntimeError("No audio was captured.")

    audio_data = np.concatenate(frames, axis=0)
    sf.write(str(output_path), audio_data, samplerate)
    print(f"Recording saved successfully: {output_path}")


def main() -> None:
    """Run the full recording workflow."""
    base_dir = Path(__file__).resolve().parent
    data_dir = ensure_data_directory(base_dir)
    output_path = create_output_path(data_dir)

    try:
        record_audio(output_path)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
