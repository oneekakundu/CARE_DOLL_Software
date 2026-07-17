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
    """Piper TTS Engine calling the piper executable."""
    
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig, voice_profile: str = "default") -> None:
        binary = config.piper_command.split()[0] if config.piper_command else ""
        if not shutil.which(binary):
            raise FileNotFoundError(f"Piper binary '{binary}' not found on system PATH.")
            
        if not os.path.exists(config.piper_model_path):
            raise FileNotFoundError(f"Piper model file not found at '{config.piper_model_path}'.")

        cmd = [binary, "-m", config.piper_model_path, "-f", str(output_path)]
        
        # Check if voice profile adjusts speed
        profile_settings = config.voice_profiles.get(voice_profile, config.voice_profiles["default"])
        rate_mult = profile_settings.get("rate_multiplier", 1.0) * config.speech_rate
        if rate_mult != 1.0:
            length_scale = 1.0 / rate_mult
            cmd.extend(["--length_scale", f"{length_scale:.2f}"])
            
        LOGGER.info(f"Running Piper command: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            input=text,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

class WindowsNativeTTSEngine(BaseTTSEngine):
    """Windows Native Speech API (SAPI5) fallback via PowerShell."""
    
    def synthesize(self, text: str, output_path: Path, config: VoiceConfig, voice_profile: str = "default") -> None:
        profile_settings = config.voice_profiles.get(voice_profile, config.voice_profiles["default"])
        rate_mult = profile_settings.get("rate_multiplier", 1.0) * config.speech_rate
        
        # Map speech rate to SAPI range (-10 to 10). Rate multiplier of 1.0 is 0.
        sapi_rate = int((rate_mult - 1.0) * 10)
        sapi_rate = max(-10, min(10, sapi_rate))
        
        # Map volume to SAPI range (0 to 100)
        sapi_volume = int(config.volume * 100)
        sapi_volume = max(0, min(100, sapi_volume))
        
        # Escape single quotes for PowerShell
        safe_text = text.replace("'", "''").replace("\n", " ").strip()
        abs_path = os.path.abspath(output_path)
        
        ps_command = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {sapi_rate}; "
            f"$synth.Volume = {sapi_volume}; "
            f"$synth.SetOutputToWaveFile('{abs_path}'); "
            f"$synth.Speak('{safe_text}'); "
            f"$synth.Dispose()"
        )
        
        LOGGER.info(f"Running SAPI fallback synthesis to file: {abs_path}")
        subprocess.run(["powershell", "-Command", ps_command], check=True, capture_output=True)

class TTSEngineFactory:
    """Selects and loads the appropriate TTS engine based on configuration and availability."""
    
    @staticmethod
    def get_engine(config: VoiceConfig) -> BaseTTSEngine:
        if config.engine_type == "piper":
            binary = config.piper_command.split()[0] if config.piper_command else ""
            if shutil.which(binary) and os.path.exists(config.piper_model_path):
                LOGGER.info("Using primary TTS engine: Piper")
                return PiperTTSEngine()
            else:
                LOGGER.warning("Piper engine is configured but binary or model was not found. Falling back to Windows Native.")
                
        LOGGER.info("Using fallback TTS engine: Windows Native SAPI")
        return WindowsNativeTTSEngine()
