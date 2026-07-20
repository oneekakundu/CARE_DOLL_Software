"""
Unit tests for EngineSelector voice engine selection system.
"""

from pathlib import Path
import pytest
from Voice_Output.voice_profile_manager import VoiceProfileManager
from Voice_Output.engine_selector import EngineSelector, ENGINE_PIPER, ENGINE_XTTS_V2
from tests.test_voice_profile_manager import create_dummy_wav


@pytest.fixture
def profile_manager(tmp_path: Path) -> VoiceProfileManager:
    """Fixture providing a clean VoiceProfileManager using temporary path."""
    return VoiceProfileManager(base_dir=tmp_path / "voices" / "profiles")


@pytest.fixture
def engine_selector(profile_manager: VoiceProfileManager) -> EngineSelector:
    """Fixture providing an EngineSelector initialized with temporary profile manager."""
    return EngineSelector(profile_manager=profile_manager)


def test_case_1_emergency_true_personalized_ready(engine_selector: EngineSelector, profile_manager: VoiceProfileManager, caplog):
    """
    Case 1: Emergency = True, Personalized voice = Ready
    Result: Piper (Emergency mode ALWAYS uses Piper)
    """
    # Setup ready personalized profile
    profile_dir = profile_manager.create_profile("primary_user", enabled=True, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")
    assert profile_manager.is_profile_ready("primary_user") is True

    with caplog.at_level("INFO"):
        selected = engine_selector.select_engine(profile_name="primary_user", emergency=True)

    assert selected == ENGINE_PIPER
    assert engine_selector.should_use_personalized_voice("primary_user", emergency=True) is False
    assert "Emergency mode active. Selecting Piper." in caplog.text


def test_case_2_emergency_false_personalized_not_ready(engine_selector: EngineSelector, profile_manager: VoiceProfileManager, caplog):
    """
    Case 2: Emergency = False, Personalized voice = Not Ready (profile missing)
    Result: Piper
    """
    assert profile_manager.is_profile_ready("primary_user") is False

    with caplog.at_level("INFO"):
        selected = engine_selector.select_engine(profile_name="primary_user", emergency=False)

    assert selected == ENGINE_PIPER
    assert engine_selector.should_use_personalized_voice("primary_user", emergency=False) is False
    assert "Personalized profile unavailable. Selecting Piper." in caplog.text


def test_case_2_b_emergency_false_personalized_incomplete(engine_selector: EngineSelector, profile_manager: VoiceProfileManager, caplog):
    """
    Case 2b: Emergency = False, Personalized profile exists but consent_confirmed is False
    Result: Piper
    """
    profile_dir = profile_manager.create_profile("primary_user", enabled=True, consent_confirmed=False)
    create_dummy_wav(profile_dir / "reference.wav")
    assert profile_manager.is_profile_ready("primary_user") is False

    with caplog.at_level("INFO"):
        selected = engine_selector.select_engine(profile_name="primary_user", emergency=False)

    assert selected == ENGINE_PIPER
    assert engine_selector.should_use_personalized_voice("primary_user", emergency=False) is False
    assert "Personalized profile unavailable. Selecting Piper." in caplog.text


def test_case_3_emergency_false_personalized_ready(engine_selector: EngineSelector, profile_manager: VoiceProfileManager, caplog):
    """
    Case 3: Emergency = False, Personalized voice = Ready
    Result: XTTS v2
    """
    profile_dir = profile_manager.create_profile("primary_user", enabled=True, consent_confirmed=True)
    create_dummy_wav(profile_dir / "reference.wav")
    assert profile_manager.is_profile_ready("primary_user") is True

    with caplog.at_level("INFO"):
        selected = engine_selector.select_engine(profile_name="primary_user", emergency=False)

    assert selected == ENGINE_XTTS_V2
    assert engine_selector.should_use_personalized_voice("primary_user", emergency=False) is True
    assert "Personalized profile ready. XTTS v2 selected." in caplog.text


def test_default_initialization():
    """Verify default EngineSelector creates VoiceProfileManager if not passed."""
    selector = EngineSelector()
    assert isinstance(selector.profile_manager, VoiceProfileManager)
