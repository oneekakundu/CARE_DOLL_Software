from .pipeline import speak, validate_setup_diagnostics
from .config import VoiceConfig, DEFAULT_VOICE_CONFIG
from .voice_profile_manager import (
    VoiceProfileManager,
    VoiceProfileError,
    ProfileNotFoundError,
    ProfileInvalidError,
    ProfileReadError,
)
from .engine_selector import EngineSelector, ENGINE_PIPER, ENGINE_XTTS_V2
from .xtts_engine import (
    XTTSv2Engine,
    XTTSEngineError,
    XTTSEngineUnavailableError,
    XTTSGenerationError,
)

__all__ = [
    "speak",
    "validate_setup_diagnostics",
    "VoiceConfig",
    "DEFAULT_VOICE_CONFIG",
    "VoiceProfileManager",
    "VoiceProfileError",
    "ProfileNotFoundError",
    "ProfileInvalidError",
    "ProfileReadError",
    "EngineSelector",
    "ENGINE_PIPER",
    "ENGINE_XTTS_V2",
    "XTTSv2Engine",
    "XTTSEngineError",
    "XTTSEngineUnavailableError",
    "XTTSGenerationError",
]

