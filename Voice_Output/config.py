from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class VoiceConfig:
    """Configuration class containing paths and parameters for Piper TTS."""
    piper_binary: Path = PROJECT_ROOT / "piper" / "piper.exe"
    voice_model: Path = PROJECT_ROOT / "models" / "en_US-lessac-medium.onnx"
    voice_config: Path = PROJECT_ROOT / "models" / "en_US-lessac-medium.onnx.json"
    output_dir: Path = PROJECT_ROOT / "data" / "tts"
    
    speech_rate: float = 1.0  # Rate multiplier
    volume: float = 1.0  # Volume multiplier (for future software gain adjustment)
    auto_play: bool = True
    save_audio: bool = True
    playback_device: str | int | None = None  # None uses default system device
    
    # Future ready voice profiles
    voice_profiles: dict[str, dict] = field(default_factory=lambda: {
        "default": {"rate_multiplier": 1.0, "style": "normal"},
        "calm": {"rate_multiplier": 0.85, "style": "calm"},
        "emergency": {"rate_multiplier": 1.15, "style": "urgent"},
        "companion": {"rate_multiplier": 1.0, "style": "friendly"}
    })
    
    # XTTS v2 Configuration
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_model_dir: Path = PROJECT_ROOT / "models" / "xtts_v2"


DEFAULT_VOICE_CONFIG = VoiceConfig()
