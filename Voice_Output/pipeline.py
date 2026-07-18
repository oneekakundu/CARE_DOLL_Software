import time
from datetime import datetime
from pathlib import Path
from .config import DEFAULT_VOICE_CONFIG, VoiceConfig
from .logger import get_voice_logger
from .audio_player import AudioPlayer
from .utils import clean_text_for_speech

LOGGER = get_voice_logger()


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
        # validate_setup checks binary, model, and config
        resolved_binary = PiperTTSEngine.validate_setup(config)
        print("✓ Found")
    except Exception as exc:
        print("✗ Failed")
        raise
        
    # 2. Check voice model
    print("Checking voice model...")
    if not Path(config.voice_model).exists():
        print("✗ Failed")
        # validate_setup would have failed first, but we keep this as safeguard
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


def speak(text: str, voice_profile: str = "default", config: VoiceConfig = DEFAULT_VOICE_CONFIG) -> Path | None:
    """
    Main entry point for generating speech from text.
    Handles cleaning, engine selection, synthesis, file saving, and playback.
    Catches errors gracefully to prevent pipeline crashes.
    """
    if not text or not text.strip():
        LOGGER.warning("Speak called with empty text. Skipping.")
        return None
        
    cleaned_text = clean_text_for_speech(text)
    
    LOGGER.info(f"Speech generation started. Engine: piper | Characters: {len(cleaned_text)} | Profile: {voice_profile}")
    start_time = time.perf_counter()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"response_{timestamp}.wav"
    output_path = config.output_dir / output_filename
    
    # Ensure directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from .tts_engine import PiperTTSEngine
        engine = PiperTTSEngine()
        engine.synthesize(cleaned_text, output_path, config, voice_profile)
        
        generation_time = time.perf_counter() - start_time
        LOGGER.info(f"Speech generation finished. Saved to: {output_path} | Generation Time: {generation_time:.2f} sec")
        
        # Audio Playback
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
                
        return output_path
        
    except Exception as exc:
        LOGGER.error(f"Voice Output Pipeline Failed: {exc}", exc_info=True)
        # Log failure and print answer to allow the application to keep running gracefully
        print(f"\n[Voice synthesis failed: {exc}]\n")
        return None
