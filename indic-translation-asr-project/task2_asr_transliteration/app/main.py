"""
Main Entry Point for Tamil ASR Transcription & Transliteration System.

This module serves as the application entry point, initializing all
components and launching the Gradio web interface for user interaction.
"""

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Global model cache - initialized once at startup
GLOBAL_MODEL_CACHE: Dict[str, Any] = {
    "asr_pipelines": {},  # model_size -> ASRPipeline instance
    "transliterator": None,  # Shared TransliterationEngine instance
}

# Thread-safe response mapping for async processing
# Maps job_id -> {"status": str, "result": Optional[Dict], "event": threading.Event}
ASYNC_RESPONSES: Dict[str, Dict[str, Any]] = {}
ASYNC_RESPONSES_LOCK = threading.Lock()

# Background worker control
_worker_thread: Optional[threading.Thread] = None
_worker_stop_event = threading.Event()


def audio_worker(
    buffer_manager,
    asr_cache: Dict,
    transliterator
) -> None:
    """
    Background daemon thread worker for asynchronous audio processing.
    
    This function continuously polls the AudioBufferManager for incoming
    audio chunks and processes them through the globally cached ASRPipeline
    and TransliterationEngine. Results are stored in a thread-safe response
    dictionary for retrieval by the Gradio interface.
    
    Args:
        buffer_manager: AudioBufferManager instance for dequeuing audio
        asr_cache: Dictionary mapping model_size to ASRPipeline instances
        transliterator: Shared TransliterationEngine instance
    
    Note:
        This function runs indefinitely until _worker_stop_event is set.
        All exceptions are caught and logged to prevent silent crashes.
    """
    logger.info("Audio worker thread started")
    
    while not _worker_stop_event.is_set():
        try:
            # Poll for audio chunk with timeout
            audio_data = buffer_manager.dequeue(timeout=1.0)
            
            if audio_data is None:
                # No audio available, continue polling
                continue
            
            # Extract job metadata from audio_data if it's a tuple
            if isinstance(audio_data, tuple) and len(audio_data) == 3:
                audio_array, job_id, target_scheme = audio_data
            else:
                # Fallback for direct array (shouldn't happen in new implementation)
                logger.warning("Received raw array instead of tuple, skipping")
                continue
            
            logger.info(f"Processing job {job_id} with audio shape {audio_array.shape}")
            
            # Determine model size from job_id prefix or default to "small"
            model_size = "small"
            if job_id.startswith("small_"):
                model_size = "small"
            elif job_id.startswith("medium_"):
                model_size = "medium"
            
            # Get ASR pipeline from cache
            asr_pipeline = asr_cache.get(model_size)
            if asr_pipeline is None:
                logger.error(f"ASR pipeline for model '{model_size}' not found in cache")
                with ASYNC_RESPONSES_LOCK:
                    if job_id in ASYNC_RESPONSES:
                        ASYNC_RESPONSES[job_id]["status"] = "error"
                        ASYNC_RESPONSES[job_id]["result"] = {"error": f"Model '{model_size}' not available"}
                        ASYNC_RESPONSES[job_id]["event"].set()
                continue
            
            # Transcribe audio
            transcription_result = asr_pipeline.transcribe(audio_array)
            transcript = transcription_result.get("text", "")
            confidence = transcription_result.get("confidence", 0.0)
            
            # Transliterate
            transliterated = ""
            if transcript:
                transliterated = transliterator.transliterate(transcript)
            
            # Build result
            result = {
                "status": "success" if transcript else "no_speech",
                "transcript": transcript,
                "transliteration": transliterated,
                "metadata": {
                    "job_id": job_id,
                    "model_size": model_size,
                    "target_scheme": target_scheme,
                    "confidence": round(confidence, 4),
                    "audio_duration_seconds": round(len(audio_array) / 16000, 2),
                    "transcript_length": len(transcript),
                    "transliteration_length": len(transliterated)
                }
            }
            
            # Store result in thread-safe response dictionary
            with ASYNC_RESPONSES_LOCK:
                if job_id in ASYNC_RESPONSES:
                    ASYNC_RESPONSES[job_id]["status"] = "completed"
                    ASYNC_RESPONSES[job_id]["result"] = result
                    ASYNC_RESPONSES[job_id]["event"].set()
                else:
                    logger.warning(f"Job {job_id} not found in response map")
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Audio worker error: {e}", exc_info=True)
            # Prevent silent crash - continue processing
            continue
    
    logger.info("Audio worker thread stopped")


def start_background_worker(buffer_manager) -> None:
    """
    Start the background audio processing worker thread.
    
    Args:
        buffer_manager: AudioBufferManager instance for the worker to poll
    
    Note:
        This function should be called once at application startup.
    """
    global _worker_thread, _worker_stop_event
    
    _worker_stop_event.clear()
    _worker_thread = threading.Thread(
        target=audio_worker,
        args=(
            buffer_manager,
            GLOBAL_MODEL_CACHE["asr_pipelines"],
            GLOBAL_MODEL_CACHE["transliterator"]
        ),
        daemon=True,
        name="AudioWorker"
    )
    _worker_thread.start()
    logger.info("Background audio worker thread started")


def stop_background_worker() -> None:
    """
    Stop the background audio processing worker thread.
    
    Note:
        This function should be called during application shutdown.
    """
    global _worker_stop_event, _worker_thread
    
    _worker_stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=5.0)
        logger.info("Background audio worker thread stopped")


def initialize_global_models() -> None:
    """
    Initialize all heavy models once at startup and cache them globally.
    
    This function pre-loads Whisper models for both "small" and "medium"
    variants, as well as the shared TransliterationEngine. Models are
    stored in GLOBAL_MODEL_CACHE for reuse across all requests.
    
    Raises:
        RuntimeError: If model initialization fails
    """
    from .asr_pipeline import ASRPipeline
    from .transliteration import TransliterationEngine
    
    logger.info("Initializing global model cache...")
    
    # Initialize ASR pipelines for each model size
    for model_size in ["small", "medium"]:
        try:
            logger.info(f"Loading Whisper '{model_size}' model...")
            asr_pipeline = ASRPipeline(model_size=model_size)
            GLOBAL_MODEL_CACHE["asr_pipelines"][model_size] = asr_pipeline
            logger.info(f"Whisper '{model_size}' model loaded and cached")
        except Exception as e:
            logger.error(f"Failed to load Whisper '{model_size}' model: {e}")
            raise RuntimeError(f"Failed to initialize ASR pipeline for '{model_size}': {e}")
    
    # Initialize shared transliteration engine
    try:
        transliterator = TransliterationEngine(
            source_scheme="Tamil",
            target_scheme="ITRANS"
        )
        GLOBAL_MODEL_CACHE["transliterator"] = transliterator
        logger.info("TransliterationEngine initialized and cached")
    except Exception as e:
        logger.error(f"Failed to initialize TransliterationEngine: {e}")
        raise RuntimeError(f"Failed to initialize transliteration engine: {e}")
    
    logger.info("Global model cache initialization complete")


def main() -> None:
    """
    Initialize components and launch the Gradio interface.
    
    This function performs the following steps:
    1. Logs application startup
    2. Initializes global model cache (heavy models loaded once)
    3. Starts background audio worker thread
    4. Imports application components with error handling
    5. Builds the Gradio interface with pre-loaded models
    6. Launches the web server on port 7860
    
    Raises:
        SystemExit: If critical dependencies are missing
    """
    logger.info("=" * 60)
    logger.info("Tamil ASR Transcription & Transliteration System")
    logger.info("=" * 60)
    
    try:
        # Import application components
        from .buffer_manager import AudioBufferManager
        from .interface import build_interface
        
        logger.info("All modules imported successfully")
        
        # Initialize global model cache (loads heavy models once)
        initialize_global_models()
        
        # Initialize buffer manager
        buffer_manager = AudioBufferManager(maxsize=10)
        logger.info(f"AudioBufferManager initialized: maxsize={buffer_manager.size()}")
        
        # Start background worker thread
        start_background_worker(buffer_manager)
        
        # Build the Gradio interface with pre-loaded models
        logger.info("Building Gradio interface with cached models...")
        demo = build_interface(
            buffer_manager=buffer_manager,
            model_cache=GLOBAL_MODEL_CACHE,
            async_responses=ASYNC_RESPONSES,
            async_responses_lock=ASYNC_RESPONSES_LOCK
        )
        
        # Launch the web server
        logger.info("Launching web interface on http://0.0.0.0:7860")
        logger.info("=" * 60)
        
        demo.launch(
            server_port=7860,
            server_name="0.0.0.0",
            share=False,
            show_error=True
        )
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error(
            "Please install required packages:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Model initialization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
