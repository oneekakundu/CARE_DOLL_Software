"""
Integration tests for Voice Output pipeline engine selection and synthesis.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from Voice_Output.config import VoiceConfig
from Voice_Output.pipeline import speak
from Voice_Output.voice_profile_manager import VoiceProfileManager
from tests.test_voice_profile_manager import create_dummy_wav


@pytest.fixture
def test_config(tmp_path: Path) -> VoiceConfig:
    """Fixture providing VoiceConfig with auto_play=False for testing."""
    return VoiceConfig(
        output_dir=tmp_path / "tts_out",
        auto_play=False,
        save_audio=True,
    )


def test_speak_backwards_compatibility(test_config: VoiceConfig, monkeypatch):
    """Verify speak(text) signature maintains 100% backwards compatibility."""
    mock_piper = MagicMock()
    monkeypatch.setattr("Voice_Output.tts_engine.PiperTTSEngine.synthesize", mock_piper)

    # Calling speak(text) without voice profile when no ready profile exists uses Piper
    out_path = speak("Default response text.", speaker_profile="unready_user", config=test_config)
    assert out_path is not None
    assert "piper" in out_path.name.lower()
    assert mock_piper.called


def test_speak_emergency_mode_routing(test_config: VoiceConfig, monkeypatch):
    """Verify speak(text, voice_profile='emergency') routes to Piper engine."""
    mock_piper = MagicMock()
    monkeypatch.setattr("Voice_Output.tts_engine.PiperTTSEngine.synthesize", mock_piper)

    out_path = speak("Immediate safety instruction!", voice_profile="emergency", config=test_config)
    assert out_path is not None
    assert "piper" in out_path.name.lower()
    assert mock_piper.called


def test_speak_companion_unready_profile_routing(test_config: VoiceConfig, monkeypatch):
    """Verify companion mode routes to Piper when profile is missing or unready."""
    mock_piper = MagicMock()
    monkeypatch.setattr("Voice_Output.tts_engine.PiperTTSEngine.synthesize", mock_piper)

    out_path = speak(
        "Hello there",
        voice_profile="companion",
        speaker_profile="unready_user",
        config=test_config,
    )
    assert out_path is not None
    assert "piper" in out_path.name.lower()
    assert mock_piper.called


def test_speak_companion_ready_profile_routing(test_config: VoiceConfig, tmp_path: Path, monkeypatch):
    """Verify companion mode routes to XTTS v2 when personalized profile is ready."""
    # Setup ready primary_user profile in temp directory
    profiles_dir = tmp_path / "voices" / "profiles"
    pm = VoiceProfileManager(base_dir=profiles_dir)
    profile_dir = pm.create_profile("primary_user", enabled=True, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")

    # Patch pipeline global managers
    monkeypatch.setattr("Voice_Output.pipeline._PROFILE_MANAGER", pm)
    monkeypatch.setattr("Voice_Output.pipeline._ENGINE_SELECTOR.profile_manager", pm)

    # Patch XTTSv2Engine
    mock_xtts_synthesize = MagicMock(side_effect=lambda text, profile_name, output_path, **kwargs: create_dummy_wav(Path(output_path)))
    monkeypatch.setattr("Voice_Output.xtts_engine.XTTSv2Engine.is_available", staticmethod(lambda: True))
    monkeypatch.setattr("Voice_Output.xtts_engine.XTTSv2Engine.synthesize", mock_xtts_synthesize)

    out_path = speak(
        "Good morning!",
        voice_profile="companion",
        speaker_profile="primary_user",
        config=test_config,
    )

    assert out_path is not None
    assert "xtts_v2" in out_path.name.lower()
    assert mock_xtts_synthesize.called


def test_speak_xtts_fallback_to_piper(test_config: VoiceConfig, tmp_path: Path, monkeypatch):
    """Verify fallback to Piper when XTTS synthesis raises an unexpected exception."""
    profiles_dir = tmp_path / "voices" / "profiles"
    pm = VoiceProfileManager(base_dir=profiles_dir)
    profile_dir = pm.create_profile("primary_user", enabled=True, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")

    monkeypatch.setattr("Voice_Output.pipeline._PROFILE_MANAGER", pm)
    monkeypatch.setattr("Voice_Output.pipeline._ENGINE_SELECTOR.profile_manager", pm)

    # Mock XTTS engine to raise exception during synthesis
    monkeypatch.setattr("Voice_Output.xtts_engine.XTTSv2Engine.is_available", staticmethod(lambda: True))
    monkeypatch.setattr("Voice_Output.xtts_engine.XTTSv2Engine.synthesize", MagicMock(side_effect=RuntimeError("GPU OOM")))

    # Mock Piper engine to succeed
    mock_piper = MagicMock()
    monkeypatch.setattr("Voice_Output.tts_engine.PiperTTSEngine.synthesize", mock_piper)

    out_path = speak(
        "Falling back cleanly",
        voice_profile="companion",
        speaker_profile="primary_user",
        config=test_config,
    )

    assert out_path is not None
    assert mock_piper.called
