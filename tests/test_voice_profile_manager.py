"""
Tests for VoiceProfileManager readiness detection and profile management.
"""

import json
import wave
from pathlib import Path

import pytest

from Voice_Output.voice_profile_manager import (
    VoiceProfileManager,
    ProfileNotFoundError,
    ProfileInvalidError,
    ProfileReadError,
)


def create_dummy_wav(file_path: Path, duration_sec: float = 0.5) -> Path:
    """Helper function to create a valid WAV file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_samples)
    return file_path


@pytest.fixture
def manager(tmp_path: Path) -> VoiceProfileManager:
    """Fixture providing a VoiceProfileManager instance with temporary directory."""
    profiles_dir = tmp_path / "voices" / "profiles"
    return VoiceProfileManager(base_dir=profiles_dir)


def test_profile_exists(manager: VoiceProfileManager):
    assert not manager.profile_exists("non_existent_profile")
    manager.get_profile_dir("test_user").mkdir(parents=True, exist_ok=True)
    assert manager.profile_exists("test_user")


def test_create_profile(manager: VoiceProfileManager):
    profile_dir = manager.create_profile(
        profile_name="user1",
        engine="xtts_v2",
        language="en",
        enabled=True,
        consent_confirmed=True,
    )
    assert profile_dir.exists()
    assert (profile_dir / "profile.json").exists()

    profile_data = manager.get_profile("user1")
    assert profile_data["profile_name"] == "user1"
    assert profile_data["engine"] == "xtts_v2"
    assert profile_data["enabled"] is True
    assert profile_data["consent_confirmed"] is True


def test_is_profile_ready_success(manager: VoiceProfileManager):
    profile_dir = manager.create_profile("ready_user", enabled=True, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")

    assert manager.is_profile_ready("ready_user") is True
    assert manager.get_profile_status("ready_user") == "READY"


def test_readiness_rule_1_missing_directory(manager: VoiceProfileManager):
    assert manager.is_profile_ready("ghost_user") is False
    assert manager.get_profile_status("ghost_user") == "DOES_NOT_EXIST"


def test_readiness_rule_2_missing_profile_json(manager: VoiceProfileManager):
    profile_dir = manager.get_profile_dir("no_json_user")
    profile_dir.mkdir(parents=True, exist_ok=True)
    create_dummy_wav(profile_dir / "reference.wav")

    assert manager.is_profile_ready("no_json_user") is False
    assert manager.get_profile_status("no_json_user") == "INCOMPLETE"


def test_readiness_rule_3_missing_reference_audio(manager: VoiceProfileManager):
    manager.create_profile("no_audio_user", enabled=True, consent_confirmed=True)

    assert manager.is_profile_ready("no_audio_user") is False
    assert manager.get_profile_status("no_audio_user") == "INCOMPLETE"


def test_readiness_rule_4_empty_or_corrupt_reference_audio(manager: VoiceProfileManager):
    profile_dir = manager.create_profile("empty_audio_user", enabled=True, consent_confirmed=True)

    # Empty audio file (0 bytes)
    empty_wav = profile_dir / "reference.wav"
    empty_wav.write_bytes(b"")

    assert manager.is_profile_ready("empty_audio_user") is False
    assert manager.get_profile_status("empty_audio_user") == "INCOMPLETE"


def test_readiness_rule_5_consent_not_confirmed(manager: VoiceProfileManager):
    profile_dir = manager.create_profile("no_consent_user", enabled=True, consent_confirmed=False)
    create_dummy_wav(profile_dir / "reference.wav")

    assert manager.is_profile_ready("no_consent_user") is False
    assert manager.get_profile_status("no_consent_user") == "INCOMPLETE"


def test_readiness_rule_6_profile_disabled(manager: VoiceProfileManager):
    profile_dir = manager.create_profile("disabled_user", enabled=False, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")

    assert manager.is_profile_ready("disabled_user") is False
    assert manager.get_profile_status("disabled_user") == "INCOMPLETE"


def test_get_profile_exceptions(manager: VoiceProfileManager):
    with pytest.raises(ProfileNotFoundError):
        manager.get_profile("non_existent")

    # Invalid JSON syntax
    profile_dir = manager.get_profile_dir("invalid_json_user")
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ProfileInvalidError):
        manager.get_profile("invalid_json_user")


def test_get_reference_audio(manager: VoiceProfileManager):
    profile_dir = manager.create_profile("ref_user")
    wav_path = create_dummy_wav(profile_dir / "reference.wav")

    ref_path = manager.get_reference_audio("ref_user")
    assert ref_path == wav_path

    # Test error when missing audio
    wav_path.unlink()
    with pytest.raises(ProfileNotFoundError):
        manager.get_reference_audio("ref_user")


def test_default_primary_user_profile():
    """Test the default workspace primary_user profile stored in data/voices/profiles/primary_user."""
    default_manager = VoiceProfileManager()
    assert default_manager.profile_exists("primary_user") is True
    assert default_manager.is_profile_ready("primary_user") is True
    assert default_manager.get_profile_status("primary_user") == "READY"
    ref_audio = default_manager.get_reference_audio("primary_user")
    assert ref_audio.exists()
    assert ref_audio.name == "reference.wav"
