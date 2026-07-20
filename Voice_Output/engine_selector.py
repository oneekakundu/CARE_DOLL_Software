"""
Voice Engine Selector for CARE DOLL Emergency Assistant.

Determines whether to use Piper TTS or XTTS v2 based on:
1. Emergency status (Emergency mode ALWAYS uses Piper)
2. Voice profile request
3. Personalized voice profile readiness
"""

from __future__ import annotations

import logging
from typing import Final

from .logger import get_voice_logger
from .voice_profile_manager import VoiceProfileManager

logger: logging.Logger = get_voice_logger()

# Engine Constants
ENGINE_PIPER: Final[str] = "piper"
ENGINE_XTTS_V2: Final[str] = "xtts_v2"


class EngineSelector:
    """
    Selects the appropriate TTS engine based on emergency status and voice profile readiness.
    """

    def __init__(self, profile_manager: VoiceProfileManager | None = None) -> None:
        """
        Initialize EngineSelector.

        :param profile_manager: VoiceProfileManager instance. Defaults to new VoiceProfileManager instance.
        """
        self.profile_manager = profile_manager if profile_manager is not None else VoiceProfileManager()

    def should_use_personalized_voice(self, profile_name: str = "primary_user", emergency: bool = False) -> bool:
        """
        Determine if personalized voice (XTTS v2) should be used.

        Personalized voice is used only if:
        1. Emergency mode is False
        2. Personalized voice profile is ready

        :param profile_name: Name of the voice profile (e.g. 'primary_user').
        :param emergency: Whether the current response is an emergency response.
        :return: True if personalized voice should be used, False otherwise.
        """
        if emergency:
            return False

        return self.profile_manager.is_profile_ready(profile_name)

    def select_engine(self, profile_name: str = "primary_user", emergency: bool = False) -> str:
        """
        Select the TTS engine to use ('piper' or 'xtts_v2').

        Decision logic:
        - If emergency: select Piper (Emergency mode ALWAYS uses Piper).
        - Else if personalized voice profile is ready: select XTTS v2.
        - Else: select Piper.

        :param profile_name: Name of the requested voice profile.
        :param emergency: Whether emergency mode is active.
        :return: String engine name ('piper' or 'xtts_v2').
        """
        if emergency:
            logger.info("Emergency mode active. Selecting Piper.")
            return ENGINE_PIPER

        if self.should_use_personalized_voice(profile_name=profile_name, emergency=False):
            logger.info("Personalized profile ready. XTTS v2 selected.")
            return ENGINE_XTTS_V2

        logger.info("Personalized profile unavailable. Selecting Piper.")
        return ENGINE_PIPER
