from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess

LOGGER = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """Uses faster-whisper model to transcribe audio files offline."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None

    def check_available(self) -> bool:
        """Checks if faster-whisper package is importable and model size is valid."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError as error:
            LOGGER.warning("faster-whisper package is not available: %s", error)
            return False

    def transcribe(self, audio_path: Path) -> str:
        """Transcribes an audio file to text. Loads the model on first call."""
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            if self._model is None:
                LOGGER.info("Loading Whisper model: '%s' on CPU", self._model_size)
                from faster_whisper import WhisperModel
                # Run on CPU with int8 quantization for offline/low-resource environments
                self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
                LOGGER.info("Whisper model loaded successfully.")

            LOGGER.info("Transcribing audio file: '%s'", audio_path.name)
            segments, info = self._model.transcribe(str(audio_path))
            LOGGER.info("Detected language: '%s' with probability %.2f", info.language, info.language_probability)
            
            text = " ".join(segment.text.strip() for segment in segments).strip()
            LOGGER.info("Transcription completed.")
            return text
        except Exception as error:
            LOGGER.error("Audio transcription failed: %s", error)
            raise RuntimeError(f"Audio transcription failed: {error}") from error


class OpenWakeWordDetector:
    """Detects custom wake-words in audio streams."""

    def __init__(self) -> None:
        self._model = None

    def check_available(self) -> bool:
        """Checks if openwakeword package is importable."""
        try:
            from openwakeword.model import Model  # noqa: F401
            return True
        except ImportError as error:
            LOGGER.warning("openwakeword package is not available: %s", error)
            return False

    def load(self) -> None:
        """Loads openwakeword model."""
        try:
            from openwakeword.model import Model
            self._model = Model()
            LOGGER.info("OpenWakeWord model loaded successfully.")
        except Exception as error:
            LOGGER.error("Failed to load OpenWakeWord model: %s", error)
            raise RuntimeError(f"Failed to load OpenWakeWord model: {error}") from error

    def score(self, audio_frame: object) -> float:
        """Calculates wake-word activation scores for an audio frame."""
        if self._model is None:
            raise RuntimeError("Wake-word detector is not loaded. Call load() first.")
        try:
            predictions = self._model.predict(audio_frame)
            return max(predictions.values(), default=0.0)
        except Exception as error:
            LOGGER.error("Wake-word score calculation failed: %s", error)
            return 0.0


class PiperSpeaker:
    """Runs the Piper Text-To-Speech binary to generate synthesized WAV audio."""

    def __init__(self, command: str) -> None:
        self._command = command

    def check_available(self) -> bool:
        """Checks if the Piper executable is found in PATH or at the specified command path."""
        # Split command to extract binary name in case arguments are passed
        binary = self._command.split()[0] if self._command else ""
        return shutil.which(binary) is not None

    def speak(self, text: str, output_path: Path) -> Path:
        """Converts text to speech wav file using Piper."""
        binary = self._command.split()[0] if self._command else ""
        if not shutil.which(binary):
            LOGGER.error("Piper executable not found at '%s'. TTS will be unavailable.", self._command)
            raise RuntimeError(
                f"Piper executable '{self._command}' was not found. Please install Piper and configure CARE_PIPER_COMMAND."
            )
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Generating TTS WAV output using Piper for text: '%s...'", text[:30])
        try:
            # Piper accepts input text via stdin and outputs WAV to file or stdout
            # If the user specified additional arguments, split them. E.g., 'piper -m model.onnx'
            cmd = self._command.split()
            if "--output_file" not in cmd:
                cmd.extend(["--output_file", str(output_path)])
                
            LOGGER.info("Executing Piper command: %s", " ".join(cmd))
            subprocess.run(
                cmd,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            LOGGER.info("TTS WAV file generated successfully at '%s'", output_path)
            return output_path
        except subprocess.CalledProcessError as error:
            stderr_output = error.stderr.decode("utf-8", errors="replace") if error.stderr else ""
            LOGGER.error("Piper process execution failed: %s. Stderr: %s", error, stderr_output)
            raise RuntimeError(f"Piper execution failed: {error}. Details: {stderr_output}") from error
        except Exception as error:
            LOGGER.error("Unexpected error during Piper TTS generation: %s", error)
            raise RuntimeError(f"TTS generation failed: {error}") from error
