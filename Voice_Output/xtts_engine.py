"""
XTTS v2 Engine for CARE DOLL Emergency Assistant.

Provides optional offline personalized voice cloning capabilities using Coqui XTTS v2.
Loaded lazily. Piper remains the primary default and emergency engine.
"""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path
from typing import Any

from .config import VoiceConfig, DEFAULT_VOICE_CONFIG
from .logger import get_voice_logger
from .voice_profile_manager import VoiceProfileManager, VoiceProfileError

logger: logging.Logger = get_voice_logger()


class XTTSEngineError(Exception):
    """Base exception class for XTTS engine errors."""
    pass


class XTTSEngineUnavailableError(XTTSEngineError):
    """Raised when XTTS v2 or required dependencies (torch/TTS) are unavailable."""
    pass


class XTTSGenerationError(XTTSEngineError):
    """Raised when XTTS v2 speech synthesis fails or generates invalid audio."""
    pass


class XTTSv2Engine:
    """
    Lazy-loaded XTTS v2 Engine for personalized voice synthesis.
    """

    def __init__(
        self,
        config: VoiceConfig | None = None,
        profile_manager: VoiceProfileManager | None = None,
    ) -> None:
        """
        Initialize the XTTSv2Engine shell without loading the model into memory.

        :param config: VoiceConfig configuration instance.
        :param profile_manager: VoiceProfileManager instance.
        """
        self.config = config if config is not None else DEFAULT_VOICE_CONFIG
        self.profile_manager = profile_manager if profile_manager is not None else VoiceProfileManager()
        self._model: Any = None
        self._device: str | None = None
        self._is_loaded: bool = False

    @staticmethod
    def is_available() -> bool:
        """
        Check whether XTTS v2 dependencies (torch and TTS) are installed and importable.

        :return: True if TTS library can be imported, False otherwise.
        """
        try:
            import torch  # noqa: F401
            import TTS.api  # noqa: F401
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"XTTS availability check failed: {e}")
            return False

    def detect_device(self) -> str:
        """
        Detect whether CUDA is available or fallback to CPU.

        :return: 'cuda' or 'cpu'.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def load_model(self, force_reload: bool = False) -> None:
        """
        Lazily load the XTTS v2 model into memory.

        :param force_reload: Force reloading model even if already loaded.
        :raises XTTSEngineUnavailableError: If dependencies or model are unavailable.
        """
        if self._is_loaded and not force_reload:
            logger.debug("XTTS v2 model is already loaded.")
            return

        if not self.is_available():
            msg = "XTTS v2 dependencies (coqui-tts / torch) are not installed or importable."
            logger.error(msg)
            raise XTTSEngineUnavailableError(msg)

        device = self.detect_device()
        self._device = device
        logger.info(f"XTTS v2 device: {device.upper()}")

        try:
            import os
            os.environ["COQUI_TOS_AGREED"] = "1"

            # Patch torchaudio.load to use soundfile backend, avoiding Windows torchcodec DLL errors
            try:
                import torchaudio
                import soundfile as sf
                import torch

                def _soundfile_load(filepath, *args, **kwargs):
                    data, sr = sf.read(str(filepath))
                    tensor = torch.from_numpy(data).float()
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.ndim == 2:
                        tensor = tensor.T
                    return tensor, sr

                torchaudio.load = _soundfile_load
            except Exception as patch_err:
                logger.debug(f"Could not patch torchaudio.load: {patch_err}")

            from TTS.api import TTS
            use_gpu = (device == "cuda")
            model_name = getattr(
                self.config,
                "xtts_model_name",
                "tts_models/multilingual/multi-dataset/xtts_v2"
            )
            logger.info(f"Loading XTTS v2 model '{model_name}' on device '{device}'...")

            # Instantiate TTS model lazily
            self._model = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)
            self._is_loaded = True
            logger.info("XTTS v2 model loaded successfully.")
        except Exception as e:
            self._is_loaded = False
            self._model = None
            msg = f"Failed to load XTTS v2 model: {e}"
            logger.error(msg)
            raise XTTSEngineUnavailableError(msg) from e

    def validate_audio_file(self, audio_path: Path) -> float:
        """
        Validate generated WAV audio output file:
        - File exists
        - File size > 0
        - Valid WAV header
        - Audio duration > 0

        :param audio_path: Path to the generated WAV file.
        :return: Duration of audio in seconds.
        :raises XTTSGenerationError: If file validation fails.
        """
        if not audio_path.exists() or not audio_path.is_file():
            raise XTTSGenerationError(f"Generated output file does not exist: {audio_path}")

        if audio_path.stat().st_size == 0:
            raise XTTSGenerationError(f"Generated output file is empty (0 bytes): {audio_path}")

        try:
            with wave.open(str(audio_path), "rb") as wave_file:
                nchannels = wave_file.getnchannels()
                sampwidth = wave_file.getsampwidth()
                framerate = wave_file.getframerate()
                nframes = wave_file.getnframes()

                if nchannels <= 0 or sampwidth <= 0 or framerate <= 0 or nframes <= 0:
                    raise XTTSGenerationError(f"Invalid WAV header parameters in file {audio_path}.")

                duration = nframes / float(framerate)
                if duration <= 0:
                    raise XTTSGenerationError(f"Generated WAV audio has zero duration in {audio_path}.")

                return duration
        except Exception as e:
            if isinstance(e, XTTSGenerationError):
                raise
            raise XTTSGenerationError(f"Invalid WAV format in generated output {audio_path}: {e}") from e

    def synthesize(
        self,
        text: str,
        profile_name: str = "primary_user",
        output_path: Path | str | None = None,
        language: str = "en",
    ) -> Path:
        """
        Synthesize speech from text using XTTS v2 and speaker reference audio.

        :param text: Text string to convert to speech.
        :param profile_name: Voice profile name containing reference.wav.
        :param output_path: Optional output path. Defaults to data/tts/response_xtts_<timestamp>.wav.
        :param language: Language code (default 'en').
        :return: Path to the validated generated WAV file.
        :raises XTTSEngineUnavailableError: If XTTS v2 cannot be loaded.
        :raises VoiceProfileError: If the voice profile is not ready.
        :raises XTTSGenerationError: If synthesis fails or output is invalid.
        """
        if not text or not text.strip():
            raise XTTSGenerationError("Synthesis text cannot be empty.")

        # Ensure model is lazily loaded
        if not self._is_loaded:
            self.load_model()

        # Get speaker reference audio from profile
        try:
            ref_audio_path = self.profile_manager.get_reference_audio(profile_name)
        except Exception as e:
            msg = f"Cannot synthesize with XTTS v2: profile '{profile_name}' reference audio missing/unreadable."
            logger.error(f"{msg}: {e}")
            raise VoiceProfileError(msg) from e

        # Determine output path
        if output_path is None:
            output_dir = self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            target_path = output_dir / f"response_xtts_{timestamp}.wav"
        else:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Synthesizing text using XTTS v2 with profile '{profile_name}' (ref: {ref_audio_path.name})...")

        try:
            # Perform synthesis with speaker_wav conditioning
            self._model.tts_to_file(
                text=text,
                file_path=str(target_path),
                speaker_wav=str(ref_audio_path),
                language=language,
            )
        except Exception as e:
            msg = f"XTTS v2 synthesis failed for profile '{profile_name}': {e}"
            logger.error(msg)
            raise XTTSGenerationError(msg) from e

        # Validate generated audio file
        duration = self.validate_audio_file(target_path)
        logger.info(f"XTTS v2 synthesis successful: {target_path} ({duration:.2f}s duration)")

        return target_path
