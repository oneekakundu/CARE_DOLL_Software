import sounddevice as sd
import soundfile as sf
from pathlib import Path
from .config import VoiceConfig
from .logger import get_voice_logger

LOGGER = get_voice_logger()

class AudioPlayer:
    """Plays WAV audio files using sounddevice and soundfile."""
    
    @staticmethod
    def play(wav_path: Path, config: VoiceConfig) -> None:
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found for playback: {wav_path}")
            
        LOGGER.info(f"Playback started: {wav_path.name}")
        data, fs = sf.read(str(wav_path))
        
        # Select device if configured
        device = config.playback_device
        
        try:
            # Play and block until complete
            sd.play(data, fs, device=device)
            sd.wait()
            LOGGER.info("Playback completed.")
        except Exception as exc:
            LOGGER.error(f"Failed to play audio: {exc}")
            raise RuntimeError(f"Audio playback failed: {exc}") from exc
