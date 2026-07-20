"""
Voice Output Pipeline for CARE DOLL Emergency Assistant.

Integrates Piper TTS, VoiceProfileManager, EngineSelector, and XTTSv2Engine.
Ensures emergency responses always use Piper while supporting personalized XTTS v2
when ready and requested.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from .audio_player import AudioPlayer
from .config import DEFAULT_VOICE_CONFIG, VoiceConfig
from .engine_selector import ENGINE_PIPER, ENGINE_XTTS_V2, EngineSelector
from .logger import get_voice_logger
from .utils import clean_text_for_speech
from .voice_profile_manager import VoiceProfileManager

LOGGER = get_voice_logger()

# Global default instances for efficient reuse
_PROFILE_MANAGER = VoiceProfileManager()
_ENGINE_SELECTOR = EngineSelector(profile_manager=_PROFILE_MANAGER)


def validate_setup_diagnostics(config: VoiceConfig = DEFAULT_VOICE_CONFIG) -> None:
    """
    Performs clear startup validation checks for Piper binary and model assets.
    Raises RuntimeError if anything is missing or invalid.
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from .tts_engine import PiperTTSEngine
    import tempfile

    # 1. Check Piper binary
    print("Checking Piper binary...")
    try:
        resolved_binary = PiperTTSEngine.validate_setup(config)
        print("✓ Found")
    except Exception as exc:
        print("✗ Failed")
        raise

    # 2. Check voice model
    print("Checking voice model...")
    if not Path(config.voice_model).exists():
        print("✗ Failed")
        raise RuntimeError(f"Voice model file not found: {config.voice_model}")
    print("✓ Found")

    # 3. Check voice config
    print("Checking config...")
    if not Path(config.voice_config).exists():
        print("✗ Failed")
        raise RuntimeError(f"Voice model config file not found: {config.voice_config}")
    print("✓ Found")

    # 4. Dry-run test synthesis
    print("Testing Piper...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_wav = Path(temp_dir) / "test_synthesis.wav"
            engine = PiperTTSEngine()
            engine.synthesize("test", test_wav, config)
        print("✓ Passed")
    except Exception as exc:
        print("✗ Failed")
        raise RuntimeError(f"Piper TTS synthesis dry-run test failed: {exc}") from exc


def speak(
    text: str,
    voice_profile: str = "default",
    speaker_profile: str = "primary_user",
    config: VoiceConfig = DEFAULT_VOICE_CONFIG,
) -> Path | None:
    """
    Main entry point for generating speech from text.

    Supports:
    - speak("Emergency warning!", voice_profile="emergency") -> Piper
    - speak("Hello", voice_profile="companion") -> Piper if no ready profile, XTTS v2 if ready
    - speak("Hello", voice_profile="companion", speaker_profile="primary_user") -> XTTS v2 if ready

    :param text: Input text to convert to speech.
    :param voice_profile: Style / mode ('emergency', 'companion', 'default', 'calm').
    :param speaker_profile: Voice profile name for personalized voice cloning ('primary_user').
    :param config: VoiceConfig instance.
    :return: Path to generated audio file, or None if save_audio is False.
    """
    if not text or not text.strip():
        LOGGER.warning("Speak called with empty text. Skipping.")
        return None

    cleaned_text = clean_text_for_speech(text)
    if not cleaned_text:
        LOGGER.warning("Cleaned text is empty after preprocessing. Skipping synthesis.")
        return None

    # Step 3: Determine whether emergency mode is active
    is_emergency = (voice_profile.lower() == "emergency")

    # Step 4: Ask EngineSelector for the correct engine
    selected_engine = _ENGINE_SELECTOR.select_engine(
        profile_name=speaker_profile,
        emergency=is_emergency,
    )

    LOGGER.info(
        f"Speech generation started. Requested Engine: {selected_engine} | "
        f"Characters: {len(cleaned_text)} | Voice Profile: {voice_profile} | "
        f"Speaker Profile: {speaker_profile} | Emergency Mode: {is_emergency}"
    )

    start_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"response_{selected_engine}_{timestamp}.wav"
    output_path = config.output_dir / output_filename

    try:
        actual_engine_used = selected_engine

        # Step 5 & 6: Validate engine availability and load XTTS lazily if selected
        if selected_engine == ENGINE_XTTS_V2:
            try:
                from .xtts_engine import XTTSv2Engine
                xtts_engine = XTTSv2Engine(config=config, profile_manager=_PROFILE_MANAGER)

                if not xtts_engine.is_available():
                    LOGGER.warning("XTTS v2 is selected but unavailable. Falling back to Piper.")
                    actual_engine_used = ENGINE_PIPER
                else:
                    # Step 7 & 8: Synthesize & Validate WAV via XTTS v2
                    output_path = xtts_engine.synthesize(
                        text=cleaned_text,
                        profile_name=speaker_profile,
                        output_path=output_path,
                    )
            except Exception as xtts_exc:
                LOGGER.error(f"XTTS v2 synthesis failed ({xtts_exc}). Falling back to Piper.")
                actual_engine_used = ENGINE_PIPER

        # Synthesize via Piper if selected or as fallback
        if actual_engine_used == ENGINE_PIPER:
            output_filename = f"response_piper_{timestamp}.wav"
            output_path = config.output_dir / output_filename
            from .tts_engine import PiperTTSEngine
            piper_engine = PiperTTSEngine()
            piper_engine.synthesize(cleaned_text, output_path, config, voice_profile)

        generation_time = time.perf_counter() - start_time
        LOGGER.info(
            f"Speech generation finished using '{actual_engine_used}'. "
            f"Saved to: {output_path} | Generation Time: {generation_time:.2f} sec"
        )

        # Step 9: Play audio
        if config.auto_play:
            AudioPlayer.play(output_path, config)

        # Clean up file if save_audio is False
        if not config.save_audio:
            try:
                output_path.unlink()
                LOGGER.info("Temporary audio file removed as save_audio is disabled.")
                return None
            except Exception as e:
                LOGGER.warning(f"Failed to remove temporary audio file: {e}")

        # Step 10: Return output path
        return output_path

    except Exception as exc:
        LOGGER.error(f"Voice Output Pipeline Failed: {exc}", exc_info=True)
        print(f"\n[Voice synthesis failed: {exc}]\n")
        return None
