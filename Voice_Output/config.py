from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class VoiceConfig:
    engine_type: str = "piper"  # "piper" or "windows_native"
    piper_command: str = "piper"  # Command or path to piper executable
    piper_model_path: str = str(PROJECT_ROOT / "models" / "en_US-lessac-medium.onnx") # Default voice model path
    speech_rate: float = 1.0  # Rate multiplier
    volume: float = 1.0  # Volume from 0.0 to 1.0
    output_dir: Path = PROJECT_ROOT / "data" / "tts"
    auto_play: bool = True
    save_audio: bool = True
    playback_device: str | int | None = None  # None uses default system device
    
    # Future ready voice profiles
    voice_profiles: dict[str, dict] = field(default_factory=lambda: {
        "default": {"rate_multiplier": 1.0, "pitch": 1.0, "style": "normal"},
        "calm": {"rate_multiplier": 0.85, "pitch": 0.95, "style": "calm"},
        "emergency": {"rate_multiplier": 1.15, "pitch": 1.05, "style": "urgent"},
        "companion": {"rate_multiplier": 1.0, "pitch": 1.02, "style": "friendly"}
    })

DEFAULT_VOICE_CONFIG = VoiceConfig()
