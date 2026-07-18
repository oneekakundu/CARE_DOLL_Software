import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from .config import VoiceConfig
from .logger import get_voice_logger

LOGGER = get_voice_logger()


class BaseTTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig, voice_profile: str = "default") -> None:
        """Synthesize text and save to the output WAV path."""
        pass


class PiperTTSEngine(BaseTTSEngine):
    """Piper TTS Engine calling the local piper executable."""

    def synthesize(self, text: str, output_path: Path, config: VoiceConfig, voice_profile: str = "default") -> None:
        # Validate paths on each synthesis run
        self.validate_setup(config)

        # Resolve binary path
        binary_str = str(config.piper_binary)
        binary = shutil.which(binary_str) or binary_str

        # Build execution command
        cmd = [binary, "-m", str(config.voice_model), "-f", str(output_path)]

        # Apply voice profile rate modifier if specified
        profile_settings = config.voice_profiles.get(voice_profile, config.voice_profiles["default"])
        rate_mult = profile_settings.get("rate_multiplier", 1.0) * config.speech_rate
        if rate_mult != 1.0:
            length_scale = 1.0 / rate_mult
            cmd.extend(["--length_scale", f"{length_scale:.2f}"])

        LOGGER.info("Loading Piper...")
        LOGGER.info(f"Binary: {binary}")
        LOGGER.info(f"Voice: {config.voice_model}")
        LOGGER.info(f"Output: {output_path}")
        LOGGER.info("Synthesizing...")

        try:
            # Run synthesis
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Validate output WAV file
            self.validate_wav(output_path)

        except subprocess.CalledProcessError as err:
            stderr_output = err.stderr.decode("utf-8", errors="replace") if err.stderr else str(err)
            LOGGER.error(f"Piper process execution failed: {err}. Details: {stderr_output}")
            raise RuntimeError(f"Piper execution failed: {err}. Details: {stderr_output}") from err
        except Exception as exc:
            LOGGER.error(f"Unexpected error during Piper TTS generation: {exc}")
            raise RuntimeError(f"TTS generation failed: {exc}") from exc

    @staticmethod
    def validate_setup(config: VoiceConfig) -> Path:
        """
        Validates that all Piper components are correctly configured and present.
        Returns the resolved Path of the Piper binary if successful, else raises RuntimeError.
        """
        # 1. Check Piper binary
        binary_path = Path(config.piper_binary)
        resolved_binary = None

        if binary_path.is_absolute():
            if binary_path.exists():
                resolved_binary = binary_path
        else:
            # Try to resolve relative to project root
            local_path = Path(__file__).resolve().parent.parent / binary_path
            if local_path.exists():
                resolved_binary = local_path
            else:
                # Try system PATH
                system_path = shutil.which(str(binary_path))
                if system_path:
                    resolved_binary = Path(system_path)

        if not resolved_binary:
            expected = str(config.piper_binary)
            raise RuntimeError(
                f"Piper TTS binary is missing.\n"
                f"Expected Location: {expected} (or on system PATH)\n"
                f"Actual Location: None\n"
                f"Please download Piper from https://github.com/rhasspy/piper/releases and place it in the expected location."
            )

        if not resolved_binary.is_file():
            raise RuntimeError(
                f"Piper TTS binary path does not point to a valid file.\n"
                f"Expected Location: {resolved_binary}\n"
                f"Actual Location: {resolved_binary} (isDirectory={resolved_binary.is_dir()})"
            )

        # Check executable permissions
        if not os.access(resolved_binary, os.X_OK):
            raise RuntimeError(
                f"Piper TTS binary lacks execute permissions.\n"
                f"Expected Location: {resolved_binary}\n"
                f"Actual Location: {resolved_binary} (missing execution rights)"
            )

        # 2. Check Voice Model
        model_path = Path(config.voice_model)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent.parent / model_path

        if not model_path.exists():
            expected = str(config.voice_model)
            raise RuntimeError(
                f"Piper voice model file (.onnx) is missing.\n"
                f"Expected Location: {expected}\n"
                f"Actual Location: None\n"
                f"Please download the voice model file (.onnx) and place it in the expected location."
            )

        # 3. Check Voice Model Config
        model_config_path = Path(config.voice_config)
        if not model_config_path.is_absolute():
            model_config_path = Path(__file__).resolve().parent.parent / model_config_path

        if not model_config_path.exists():
            expected = str(config.voice_config)
            raise RuntimeError(
                f"Piper voice model configuration file (.json) is missing.\n"
                f"Expected Location: {expected}\n"
                f"Actual Location: None\n"
                f"Please download the model config file (.json) and place it in the expected location."
            )

        return resolved_binary

    @staticmethod
    def validate_wav(path: Path) -> None:
        """Automatically verifies that the generated output is a valid non-empty WAV file."""
        if not path.exists():
            raise FileNotFoundError(f"Generated WAV file is missing: {path}")
            
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Generated WAV file is empty (0 bytes): {path}")

        # Check WAV RIFF and WAVE header markers
        with path.open("rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise ValueError(f"Generated WAV header is truncated (got only {len(header)} bytes): {path}")
            if header[0:4] != b"RIFF":
                raise ValueError(f"Invalid WAV format: missing 'RIFF' marker (got {header[0:4]}): {path}")
            if header[8:12] != b"WAVE":
                raise ValueError(f"Invalid WAV format: missing 'WAVE' marker (got {header[8:12]}): {path}")
