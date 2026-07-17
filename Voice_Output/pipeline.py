import time
from datetime import datetime
from pathlib import Path
from .config import DEFAULT_VOICE_CONFIG, VoiceConfig
from .logger import get_voice_logger
from .tts_engine import TTSEngineFactory
from .audio_player import AudioPlayer
from .utils import clean_text_for_speech

LOGGER = get_voice_logger()

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
    
    # Log synthesis start
    LOGGER.info(f"Speech generation started. Engine: {config.engine_type} | Characters: {len(cleaned_text)} | Profile: {voice_profile}")
    start_time = time.perf_counter()
    
    # Determine output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"response_{timestamp}.wav"
    output_path = config.output_dir / output_filename
    
    # Ensure directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load engine and synthesize
        engine = TTSEngineFactory.get_engine(config)
        engine.synthesize(cleaned_text, output_path, config, voice_profile)
        
        generation_time = time.perf_counter() - start_time
        LOGGER.info(f"Speech generation finished. Saved to: {output_path} | Generation Time: {generation_time:.2f} sec")
        
        # Audio Playback
        if config.auto_play:
            playback_start = time.perf_counter()
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
