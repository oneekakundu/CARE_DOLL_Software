"""
Unit tests for XTTSv2Engine optional offline voice cloning component.
"""

import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from Voice_Output.voice_profile_manager import VoiceProfileManager, VoiceProfileError
from Voice_Output.xtts_engine import (
    XTTSv2Engine,
    XTTSEngineUnavailableError,
    XTTSGenerationError,
)
from tests.test_voice_profile_manager import create_dummy_wav


@pytest.fixture
def profile_manager(tmp_path: Path) -> VoiceProfileManager:
    """Fixture providing a clean VoiceProfileManager using temporary directory."""
    return VoiceProfileManager(base_dir=tmp_path / "voices" / "profiles")


@pytest.fixture
def xtts_engine(tmp_path: Path, profile_manager: VoiceProfileManager) -> XTTSv2Engine:
    """Fixture providing an XTTSv2Engine shell instance."""
    from Voice_Output.config import VoiceConfig
    config = VoiceConfig(output_dir=tmp_path / "tts_out")
    return XTTSv2Engine(config=config, profile_manager=profile_manager)


def test_lazy_loading_on_init(xtts_engine: XTTSv2Engine):
    """Verify that creating XTTSv2Engine shell does NOT load model into memory."""
    assert xtts_engine._is_loaded is False
    assert xtts_engine._model is None
    assert xtts_engine._device is None


def test_detect_device(xtts_engine: XTTSv2Engine):
    """Verify device detection returns 'cuda' or 'cpu'."""
    device = xtts_engine.detect_device()
    assert device in ("cuda", "cpu")


def test_validate_audio_file_success(xtts_engine: XTTSv2Engine, tmp_path: Path):
    """Verify audio file validation on valid WAV file."""
    wav_path = create_dummy_wav(tmp_path / "valid_response.wav", duration_sec=1.0)
    duration = xtts_engine.validate_audio_file(wav_path)
    assert duration > 0.0


def test_validate_audio_file_failures(xtts_engine: XTTSv2Engine, tmp_path: Path):
    """Verify audio file validation raises XTTSGenerationError for invalid files."""
    # Non-existent file
    with pytest.raises(XTTSGenerationError):
        xtts_engine.validate_audio_file(tmp_path / "missing.wav")

    # Empty file (0 bytes)
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(XTTSGenerationError):
        xtts_engine.validate_audio_file(empty_file)

    # Corrupt file
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"NOT_A_WAV_HEADER_DATA")
    with pytest.raises(XTTSGenerationError):
        xtts_engine.validate_audio_file(corrupt_file)


def test_load_model_unavailable(xtts_engine: XTTSv2Engine, monkeypatch):
    """Verify load_model raises XTTSEngineUnavailableError when TTS is not available."""
    monkeypatch.setattr(XTTSv2Engine, "is_available", staticmethod(lambda: False))
    with pytest.raises(XTTSEngineUnavailableError):
        xtts_engine.load_model()


def test_synthesize_empty_text(xtts_engine: XTTSv2Engine):
    """Verify synthesize raises XTTSGenerationError when empty text is provided."""
    with pytest.raises(XTTSGenerationError):
        xtts_engine.synthesize(text="")


def test_synthesize_missing_profile(xtts_engine: XTTSv2Engine, monkeypatch):
    """Verify synthesize raises VoiceProfileError when profile reference audio is missing."""
    monkeypatch.setattr(xtts_engine, "_is_loaded", True)
    monkeypatch.setattr(xtts_engine, "_model", MagicMock())

    with pytest.raises(VoiceProfileError):
        xtts_engine.synthesize("Hello world", profile_name="non_existent_profile")


def test_synthesize_mocked_success(xtts_engine: XTTSv2Engine, profile_manager: VoiceProfileManager, tmp_path: Path, monkeypatch):
    """Verify synthesize flow end-to-end with mocked TTS model."""
    profile_dir = profile_manager.create_profile("primary_user", enabled=True, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")

    mock_tts = MagicMock()

    def mock_tts_to_file(text, file_path, speaker_wav, language):
        create_dummy_wav(Path(file_path), duration_sec=1.2)

    mock_tts.tts_to_file = mock_tts_to_file

    monkeypatch.setattr(XTTSv2Engine, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(xtts_engine, "detect_device", lambda: "cpu")
    xtts_engine._model = mock_tts
    xtts_engine._is_loaded = True

    out_file = tmp_path / "custom_out.wav"
    result_path = xtts_engine.synthesize(
        text="Hello, this is a test.",
        profile_name="primary_user",
        output_path=out_file,
    )

    assert result_path == out_file
    assert result_path.exists()
    assert result_path.stat().st_size > 0
    assert xtts_engine.validate_audio_file(result_path) > 0.0
