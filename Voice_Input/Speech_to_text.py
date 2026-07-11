#!/usr/bin/env python3
"""Convert a WAV audio file into text using an offline speech recognition library."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the CARE_DOLL_Software project root based on this file location."""
    return Path(__file__).resolve().parent.parent


def ensure_output_directories(project_root: Path) -> Path:
    """Create the text output directory if it does not already exist."""
    text_dir = project_root / "data" / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    return text_dir


def speech_to_text(audio_path: str | Path) -> Path:
    """Transcribe the provided audio file and save the text into the project data/text folder."""
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    project_root = get_project_root()
    text_dir = ensure_output_directories(project_root)
    output_path = text_dir / f"{audio_path.stem}.txt"

    try:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(audio_path), beam_size=5)
            transcript = " ".join(segment.text for segment in segments).strip()
        except ImportError:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), fp16=False)
            transcript = result.get("text", "").strip()
    except Exception as exc:
        raise RuntimeError(f"Speech-to-text failed: {exc}") from exc

    output_path.write_text(transcript, encoding="utf-8")
    return output_path.resolve()


def main(audio_path: str | Path) -> Path:
    """Run speech-to-text for the provided audio file."""
    return speech_to_text(audio_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python speech_to_text.py <audio_file>")
    main(sys.argv[1])
