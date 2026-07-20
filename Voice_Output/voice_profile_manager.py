"""
Voice Profile Management System for CARE DOLL Emergency Assistant.

Manages personalized voice profiles and checks readiness for future XTTS v2 integration.
Does NOT modify or replace the core Piper TTS engine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .logger import get_voice_logger

logger: logging.Logger = get_voice_logger()


class VoiceProfileError(Exception):
    """Base exception class for voice profile errors."""
    pass


class ProfileNotFoundError(VoiceProfileError):
    """Raised when a voice profile or profile directory does not exist."""
    pass


class ProfileInvalidError(VoiceProfileError):
    """Raised when a voice profile JSON format or metadata is invalid."""
    pass


class ProfileReadError(VoiceProfileError):
    """Raised when profile files or reference audio cannot be read."""
    pass


class VoiceProfileManager:
    """
    Manages voice profiles for personalized TTS engines (such as XTTS v2).
    
    Determines whether a profile exists, is valid, and is complete and ready for use.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        """
        Initialize the VoiceProfileManager.

        :param base_dir: Directory storing voice profiles. Defaults to project data/voices/profiles.
        """
        if base_dir is None:
            # Default path relative to project root
            project_root = Path(__file__).resolve().parent.parent
            self.base_dir = project_root / "data" / "voices" / "profiles"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"VoiceProfileManager initialized with base_dir: {self.base_dir}")

    def get_profile_dir(self, profile_name: str) -> Path:
        """Return the directory path for a given profile name."""
        return self.base_dir / profile_name

    def profile_exists(self, profile_name: str) -> bool:
        """
        Check if a profile directory exists.

        :param profile_name: Name of the profile (e.g., 'primary_user').
        :return: True if the directory exists and is a folder, False otherwise.
        """
        profile_dir = self.get_profile_dir(profile_name)
        exists = profile_dir.exists() and profile_dir.is_dir()
        logger.debug(f"profile_exists('{profile_name}'): {exists}")
        return exists

    def create_profile(
        self,
        profile_name: str,
        engine: str = "xtts_v2",
        language: str = "en",
        reference_audio: str = "reference.wav",
        enabled: bool = True,
        consent_confirmed: bool = True,
        reference_audio_data: bytes | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Create a voice profile folder and profile.json metadata file.

        :param profile_name: Name of the voice profile (e.g., 'primary_user').
        :param engine: Target TTS engine (default 'xtts_v2').
        :param language: ISO language code (default 'en').
        :param reference_audio: Relative filename of the reference audio (default 'reference.wav').
        :param enabled: Whether the profile is enabled.
        :param consent_confirmed: Whether user voice consent is confirmed.
        :param reference_audio_data: Optional bytes content or Path to copy as reference audio.
        :param overwrite: If True, existing profile files will be overwritten.
        :return: Path to the profile directory.
        """
        profile_dir = self.get_profile_dir(profile_name)
        if profile_dir.exists() and not overwrite and (profile_dir / "profile.json").exists():
            logger.info(f"Profile '{profile_name}' already exists at {profile_dir}. Skipping creation.")
            return profile_dir

        profile_dir.mkdir(parents=True, exist_ok=True)
        json_path = profile_dir / "profile.json"

        profile_metadata = {
            "profile_name": profile_name,
            "engine": engine,
            "language": language,
            "reference_audio": reference_audio,
            "enabled": enabled,
            "consent_confirmed": consent_confirmed,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(profile_metadata, f, indent=2)

        logger.info(f"Created profile.json for '{profile_name}' at {json_path}")

        # Handle reference audio creation/copying if provided
        target_audio_path = profile_dir / reference_audio
        if reference_audio_data is not None:
            if isinstance(reference_audio_data, Path):
                if reference_audio_data.exists():
                    target_audio_path.write_bytes(reference_audio_data.read_bytes())
                    logger.info(f"Copied reference audio from {reference_audio_data} to {target_audio_path}")
                else:
                    raise ProfileReadError(f"Reference audio source path does not exist: {reference_audio_data}")
            elif isinstance(reference_audio_data, bytes):
                target_audio_path.write_bytes(reference_audio_data)
                logger.info(f"Wrote reference audio bytes ({len(reference_audio_data)} bytes) to {target_audio_path}")

        return profile_dir

    def get_profile(self, profile_name: str) -> dict[str, Any]:
        """
        Load and return the profile metadata dictionary.

        :param profile_name: Name of the profile.
        :return: Dictionary containing profile metadata.
        :raises ProfileNotFoundError: If the profile directory or profile.json is missing.
        :raises ProfileInvalidError: If profile.json contains invalid JSON.
        """
        if not self.profile_exists(profile_name):
            logger.warning(f"Failed to get profile '{profile_name}': directory not found.")
            raise ProfileNotFoundError(f"Profile directory for '{profile_name}' does not exist.")

        profile_dir = self.get_profile_dir(profile_name)
        json_path = profile_dir / "profile.json"

        if not json_path.exists() or not json_path.is_file():
            logger.warning(f"Failed to get profile '{profile_name}': profile.json missing at {json_path}.")
            raise ProfileNotFoundError(f"profile.json missing for profile '{profile_name}'.")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ProfileInvalidError(f"profile.json for '{profile_name}' must be a JSON object.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in profile.json for '{profile_name}': {e}")
            raise ProfileInvalidError(f"Invalid JSON syntax in profile.json for '{profile_name}': {e}") from e
        except Exception as e:
            logger.error(f"Error reading profile.json for '{profile_name}': {e}")
            raise ProfileReadError(f"Could not read profile.json for '{profile_name}': {e}") from e

    def validate_profile(self, profile_name: str) -> tuple[bool, str]:
        """
        Validate all 6 readiness conditions for a voice profile.

        Readiness conditions:
        1. Profile directory exists.
        2. profile.json exists and is valid JSON.
        3. Reference audio file exists inside the profile directory.
        4. Reference audio file is readable and non-empty.
        5. consent_confirmed is True.
        6. enabled is True.

        :param profile_name: Name of the profile to validate.
        :return: A tuple of (is_ready: bool, status_message: str).
        """
        # Condition 1: Profile directory exists
        profile_dir = self.get_profile_dir(profile_name)
        if not profile_dir.exists() or not profile_dir.is_dir():
            msg = f"Profile directory does not exist: {profile_dir}"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        # Condition 2: profile.json exists and is valid
        json_path = profile_dir / "profile.json"
        if not json_path.exists() or not json_path.is_file():
            msg = f"profile.json does not exist in profile directory: {json_path}"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        try:
            profile_data = self.get_profile(profile_name)
        except (ProfileNotFoundError, ProfileInvalidError, ProfileReadError) as err:
            msg = f"Invalid or unreadable profile.json: {err}"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        # Check required keys in profile.json
        required_keys = {"reference_audio", "enabled", "consent_confirmed"}
        missing_keys = required_keys - set(profile_data.keys())
        if missing_keys:
            msg = f"profile.json is missing required field(s): {', '.join(sorted(missing_keys))}"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        # Condition 5: consent_confirmed is true
        if not profile_data.get("consent_confirmed", False):
            msg = f"Consent not confirmed for profile '{profile_name}' (consent_confirmed=False)"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        # Condition 6: enabled is true
        if not profile_data.get("enabled", False):
            msg = f"Profile '{profile_name}' is disabled (enabled=False)"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        # Condition 3 & 4: reference audio exists and is readable / non-empty
        ref_audio_name = profile_data.get("reference_audio", "reference.wav")
        ref_audio_path = profile_dir / ref_audio_name

        if not ref_audio_path.exists() or not ref_audio_path.is_file():
            msg = f"Reference audio file does not exist: {ref_audio_path}"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        try:
            # Check readability and non-empty status
            size = ref_audio_path.stat().st_size
            if size == 0:
                msg = f"Reference audio file is empty (0 bytes): {ref_audio_path}"
                logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
                return False, msg

            with open(ref_audio_path, "rb") as f:
                _ = f.read(10)  # Attempt to read header bytes
        except Exception as err:
            msg = f"Reference audio file cannot be read: {err}"
            logger.debug(f"validate_profile('{profile_name}'): FAIL - {msg}")
            return False, msg

        msg = f"Profile '{profile_name}' is complete and ready."
        logger.info(f"validate_profile('{profile_name}'): SUCCESS - {msg}")
        return True, msg

    def is_profile_ready(self, profile_name: str) -> bool:
        """
        Determine whether a profile is complete and ready for use.

        Returns False if any readiness condition fails.
        Never raises an exception; logs the status.

        :param profile_name: Name of the profile.
        :return: True if complete and ready, False otherwise.
        """
        is_ready, reason = self.validate_profile(profile_name)
        if not is_ready:
            logger.info(f"Profile '{profile_name}' is not ready: {reason}")
        return is_ready

    def get_profile_status(self, profile_name: str) -> str:
        """
        Distinguish the profile state into one of three distinct categories:
        - 'DOES_NOT_EXIST': Directory or profile folder does not exist.
        - 'INCOMPLETE': Folder exists, but fails readiness validation.
        - 'READY': Profile passes all readiness checks.

        :param profile_name: Name of the profile.
        :return: One of ('DOES_NOT_EXIST', 'INCOMPLETE', 'READY').
        """
        if not self.profile_exists(profile_name):
            return "DOES_NOT_EXIST"
        if self.is_profile_ready(profile_name):
            return "READY"
        return "INCOMPLETE"

    def get_reference_audio(self, profile_name: str) -> Path:
        """
        Retrieve the path to the reference audio file for a profile.

        :param profile_name: Name of the profile.
        :return: Path to the reference audio file.
        :raises ProfileNotFoundError: If profile or reference audio file does not exist.
        :raises ProfileReadError: If reference audio file is not readable.
        """
        profile_data = self.get_profile(profile_name)
        profile_dir = self.get_profile_dir(profile_name)

        ref_audio_name = profile_data.get("reference_audio", "reference.wav")
        ref_audio_path = profile_dir / ref_audio_name

        if not ref_audio_path.exists() or not ref_audio_path.is_file():
            raise ProfileNotFoundError(
                f"Reference audio file '{ref_audio_name}' missing for profile '{profile_name}' at {ref_audio_path}."
            )

        try:
            if ref_audio_path.stat().st_size == 0:
                raise ProfileReadError(f"Reference audio file '{ref_audio_path}' is empty (0 bytes).")
            with open(ref_audio_path, "rb") as f:
                _ = f.read(10)
        except Exception as e:
            if isinstance(e, ProfileReadError):
                raise
            raise ProfileReadError(f"Cannot read reference audio file '{ref_audio_path}': {e}") from e

        return ref_audio_path
