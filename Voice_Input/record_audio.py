#!/usr/bin/env python3
"""Record microphone audio and save it into the project's data/audio folder."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def get_project_root() -> Path:
    """Return the CARE_DOLL_Software project root based on this file location."""
    return Path(__file__).resolve().parent.parent


def ensure_data_directories(project_root: Path) -> Path:
    """Create the required project data folders if they do not exist."""
    audio_dir = project_root / "data" / "audio"
    (project_root / "data" / "text").mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def create_output_path(audio_dir: Path) -> Path:
    """Create a unique WAV file name using the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return audio_dir / f"recording_{timestamp}.wav"


def record_audio(samplerate: int = 44100, channels: int = 1) -> Path:
    """Record audio until the user presses Enter to stop and return the saved file path."""
    frames: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames_count: int, time_info, status) -> None:
        """Store incoming audio chunks."""
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

    project_root = get_project_root()
    audio_dir = ensure_data_directories(project_root)
    output_path = create_output_path(audio_dir)

    audio_data = np.concatenate(frames, axis=0)
    sf.write(str(output_path), audio_data, samplerate)
    print(f"Recording saved successfully: {output_path}")
    return output_path.resolve()


def main() -> None:
    """Run the recording workflow as a standalone script."""
    try:
        audio_path = record_audio()
        print(f"Saved audio file: {audio_path}")
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
