"""
Gradio Interface for ASR Transcription and Transliteration.

This module provides a web-based user interface using Gradio for the
Tamil ASR transcription and transliteration system. It includes audio
upload, recording, model selection, and output display components.
"""

import logging
import threading
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import gradio as gr

# Import application components
from .buffer_manager import AudioBufferManager
from .asr_pipeline import ASRPipeline
from .transliteration import TransliterationEngine

# Configure module logger
logger = logging.getLogger(__name__)


def build_interface(
    buffer_manager: AudioBufferManager,
    model_cache: Dict[str, Any],
    async_responses: Dict[str, Dict[str, Any]],
    async_responses_lock: threading.Lock
) -> gr.Blocks:
    """
    Build and configure the Gradio interface for ASR system.
    
    This function creates a complete Gradio Blocks application with
    audio input (upload and microphone), model configuration, and
    transcription/transliteration output displays.
    
    Args:
        buffer_manager: Shared AudioBufferManager instance for enqueueing audio
        model_cache: Global cache containing pre-loaded ASR pipelines and transliterator
        async_responses: Thread-safe dictionary mapping job IDs to results
        async_responses_lock: Lock for thread-safe access to async_responses
    
    Returns:
        Configured gr.Blocks instance ready to launch
    
    Layout:
        LEFT COLUMN:
            - Audio upload component
            - Microphone recording component
            - Whisper model size dropdown
            - Output script scheme dropdown
            - Primary action button
        
        RIGHT COLUMN:
            - ASR transcript textbox (Tamil)
            - Transliteration output textbox
            - Metadata JSON display
            - Clear button
    """
    
    def process_audio(
        audio_path: Optional[str],
        model_size: str,
        target_scheme: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process uploaded or recorded audio through ASR pipeline.
        
        This function orchestrates the complete transcription and
        transliteration workflow using the producer-consumer pattern:
        1. Preprocesses audio to extract audio array
        2. Generates a unique job ID
        3. Enqueues audio with metadata into the buffer manager
        4. Waits for the background worker to complete processing
        5. Returns results from the thread-safe response dictionary
        
        Args:
            audio_path: Path to audio file (from upload or recording)
            model_size: Whisper model size ("small" or "medium")
            target_scheme: Target transliteration scheme
        
        Returns:
            Tuple of (transcript, transliteration, metadata_dict)
            On error, returns error message in transcript field
        """
        try:
            # Validate input
            if audio_path is None:
                return (
                    "Please upload or record audio first.",
                    "",
                    {"error": "No audio provided"}
                )
            
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return (
                    f"Audio file not found: {audio_path}",
                    "",
                    {"error": "File not found", "path": str(audio_path)}
                )
            
            logger.info(f"Processing audio: {audio_path}, model: {model_size}")
            
            # Get ASR pipeline from cache for preprocessing
            asr_pipeline = model_cache["asr_pipelines"].get(model_size)
            if asr_pipeline is None:
                return (
                    f"Model '{model_size}' not available in cache.",
                    "",
                    {"error": f"Model '{model_size}' not initialized"}
                )
            
            # Preprocess audio to get numpy array
            audio_array = asr_pipeline.preprocess_audio(str(audio_path))
            
            # Generate unique job ID with model_size prefix
            job_id = f"{model_size}_{uuid.uuid4().hex[:8]}"
            
            # Create threading event for this job
            job_event = threading.Event()
            
            # Register job in async responses dictionary
            with async_responses_lock:
                async_responses[job_id] = {
                    "status": "pending",
                    "result": None,
                    "event": job_event
                }
            
            # Enqueue audio chunk with job metadata (Producer)
            # Package as tuple: (audio_array, job_id, target_scheme)
            enqueue_success = buffer_manager.enqueue((audio_array, job_id, target_scheme))
            
            if not enqueue_success:
                logger.error(f"Failed to enqueue job {job_id}")
                with async_responses_lock:
                    del async_responses[job_id]
                return (
                    "Error: Audio queue is full. Please try again.",
                    "",
                    {"error": "Queue full", "job_id": job_id}
                )
            
            logger.info(f"Job {job_id} enqueued, waiting for processing...")
            
            # Wait for background worker to complete (Consumer result)
            # Timeout after 120 seconds to prevent indefinite blocking
            completed = job_event.wait(timeout=120.0)
            
            if not completed:
                logger.error(f"Job {job_id} timed out")
                with async_responses_lock:
                    del async_responses[job_id]
                return (
                    "Error: Processing timed out. Please try again.",
                    "",
                    {"error": "Timeout", "job_id": job_id}
                )
            
            # Retrieve result from async responses
            with async_responses_lock:
                job_data = async_responses.get(job_id, {})
                job_status = job_data.get("status", "unknown")
                job_result = job_data.get("result", {})
                
                # Clean up job entry
                del async_responses[job_id]
            
            # Handle job errors
            if job_status == "error":
                error_msg = job_result.get("error", "Unknown error")
                return (
                    f"Error: {error_msg}",
                    "",
                    {"error": error_msg, "job_id": job_id}
                )
            
            # Extract results from successful job
            if job_status == "completed" and job_result:
                transcript = job_result.get("transcript", "")
                transliteration = job_result.get("transliteration", "")
                metadata = job_result.get("metadata", {})
                
                # Handle empty transcription
                if not transcript:
                    return (
                        "(No speech detected)",
                        "",
                        {
                            "status": "no_speech",
                            "confidence": 0.0,
                            "duration_seconds": len(audio_array) / 16000,
                            "job_id": job_id
                        }
                    )
                
                logger.info(f"Job {job_id} complete: {len(transcript)} chars transcribed")
                return (transcript, transliteration, metadata)
            else:
                return (
                    "Error: Unexpected job status.",
                    "",
                    {"error": f"Unexpected status: {job_status}", "job_id": job_id}
                )
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            error_metadata = {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            return (
                f"Error: {str(e)}",
                "",
                error_metadata
            )
    
    def clear_outputs() -> Tuple[str, str, Dict[str, Any]]:
        """
        Clear all output fields.
        
        Returns:
            Empty strings for text outputs and empty dict for metadata
        """
        return ("", "", {})
    
    # Create the Gradio interface using Blocks
    with gr.Blocks(
        title="Tamil ASR Transcription & Transliteration",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            """
            # 🎙️ Tamil ASR Transcription & Transliteration
            
            Upload or record Tamil speech audio to get automatic transcription 
            in Tamil script and transliteration to your chosen romanization scheme.
            
            **Supported audio formats:** WAV, MP3, OGG, FLAC, M4A, WebM
            """
        )
        
        with gr.Row(equal_height=True):
            # LEFT COLUMN - Input controls
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📥 Audio Input")
                
                # Audio upload
                audio_upload = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"],
                    interactive=True
                )
                
                # Microphone recording
                audio_mic = gr.Audio(
                    label="Record Audio (Microphone)",
                    type="filepath",
                    sources=["microphone"],
                    interactive=True
                )
                
                gr.Markdown("### ⚙️ Configuration")
                
                # Model size selection
                model_dropdown = gr.Dropdown(
                    choices=["small", "medium"],
                    value="small",
                    label="Whisper Model Size",
                    info="Small: Faster, Medium: More accurate"
                )
                
                # Target scheme selection
                scheme_dropdown = gr.Dropdown(
                    choices=["ITRANS", "HK", "IAST", "Devanagari"],
                    value="ITRANS",
                    label="Output Script Scheme",
                    info="Choose romanization or Devanagari output"
                )
                
                # Action button
                process_btn = gr.Button(
                    "🎯 Transcribe & Transliterate",
                    variant="primary",
                    size="lg"
                )
            
            # RIGHT COLUMN - Outputs
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📤 Results")
                
                # ASR Transcript
                transcript_output = gr.Textbox(
                    label="ASR Transcript (Tamil)",
                    placeholder="Tamil transcription will appear here...",
                    lines=5,
                    max_lines=10
                )
                
                # Transliteration output
                translit_output = gr.Textbox(
                    label="Transliteration Output",
                    placeholder="Transliterated text will appear here...",
                    lines=5,
                    max_lines=10
                )
                
                # Metadata display
                metadata_output = gr.JSON(
                    label="Processing Metadata",
                    visible=True
                )
                
                # Clear button
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    size="lg"
                )
        
        # Add usage examples
        gr.Markdown(
            """
            ### 📝 Usage Instructions
            
            1. **Upload** an existing audio file OR **Record** using your microphone
            2. Select the desired **Whisper model size** (small for speed, medium for accuracy)
            3. Choose your preferred **Output Script Scheme** (ITRANS recommended for most users)
            4. Click **"Transcribe & Transliterate"** to process the audio
            5. View results in the right panel along with processing metadata
            
            **Note:** Processing time depends on audio length and model size.
            """
        )
        
        # Wire up event handlers
        # Use either upload or mic input (whichever is provided)
        process_btn.click(
            fn=process_audio,
            inputs=[audio_upload, model_dropdown, scheme_dropdown],
            outputs=[transcript_output, translit_output, metadata_output]
        )
        
        # Also allow mic input to trigger processing
        audio_mic.change(
            fn=process_audio,
            inputs=[audio_mic, model_dropdown, scheme_dropdown],
            outputs=[transcript_output, translit_output, metadata_output]
        )
        
        # Clear button handler
        clear_btn.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[transcript_output, translit_output, metadata_output]
        )
    
    logger.info("Gradio interface built successfully")
    return demo
