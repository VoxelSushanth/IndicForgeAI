"""
ASR Pipeline using Whisper for Tamil Speech Recognition.

This module provides a production-grade automatic speech recognition pipeline
using OpenAI's Whisper model via the transformers library. It handles audio
preprocessing, transcription, and error management for Tamil language input.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# Configure module logger
logger = logging.getLogger(__name__)


class ASRPipeline:
    """
    Whisper-based ASR pipeline for Tamil speech transcription.
    
    This class wraps the HuggingFace transformers pipeline for automatic
    speech recognition, providing preprocessing, transcription, and error
    handling capabilities optimized for Tamil language input.
    
    Attributes:
        model_size: Size of Whisper model to use (tiny, base, small, medium, large)
        device: Compute device for model inference (cuda/cpu/mps)
        _pipeline: Internal transformers pipeline instance
    
    Example:
        >>> asr = ASRPipeline(model_size="small", device="cuda")
        >>> audio = asr.preprocess_audio("speech.wav")
        >>> result = asr.transcribe(audio)
        >>> print(result["text"])
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}
    
    def __init__(self, model_size: str = "small", device: Optional[str] = None) -> None:
        """
        Initialize the ASR pipeline with Whisper model.
        
        Args:
            model_size: Whisper model size variant (default: "small")
                       Options: tiny, base, small, medium, large, large-v2, large-v3
            device: Override for compute device. If None, auto-detects CUDA/MPS/CPU
        
        Raises:
            RuntimeError: If model fails to load after all retry attempts
        """
        self.model_size = model_size
        self.device = self._auto_detect_device() if device is None else device
        
        logger.info(f"Initializing Whisper {model_size} model on {self.device}")
        
        try:
            model_name = f"openai/whisper-{model_size}"
            
            # Load processor and model separately for better control
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Create the pipeline with chunking for long-form audio
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device if self.device != "mps" else -1,  # MPS not supported via device param
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                model_kwargs={"use_safetensors": True},
                chunk_length_s=30,  # Enable overlap-add chunking for audio > 30s
                stride_length_s=(4, 2)  # Stride: 4s left, 2s right overlap
            )
            
            logger.info(f"Successfully loaded Whisper {model_size} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Whisper model loading failed: {e}")
    
    def _auto_detect_device(self) -> str:
        """
        Auto-detect the best available compute device.
        
        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple MPS available")
            return "mps"
        else:
            logger.info("Using CPU for inference")
            return "cpu"
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file for ASR.
        
        This method loads audio from various formats, resamples to 16kHz,
        converts to mono, and normalizes amplitude to [-1, 1] range.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, .ogg, .flac, etc.)
        
        Returns:
            NumPy array of audio samples at 16kHz, normalized to [-1, 1]
        
        Raises:
            ValueError: If file format is not supported or file doesn't exist
            FileNotFoundError: If the specified audio file does not exist
        """
        path = Path(audio_path)
        
        # Check file existence
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check file extension
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            import librosa
            
            # Load audio with librosa (handles all supported formats)
            audio_array, sample_rate = librosa.load(
                str(audio_path),
                sr=16000,  # Whisper expects 16kHz
                mono=True   # Convert to mono
            )
            
            # Normalize to [-1, 1] range (librosa already does this, but ensure it)
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            # Ensure float32 dtype
            audio_array = audio_array.astype(np.float32)
            
            logger.info(
                f"Preprocessed audio: {len(audio_array)} samples, "
                f"{len(audio_array)/16000:.2f}s duration"
            )
            
            return audio_array
            
        except ImportError:
            logger.error("librosa not installed. Install with: pip install librosa")
            raise RuntimeError("librosa library is required for audio preprocessing")
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise RuntimeError(f"Audio preprocessing failed: {e}")
    
    def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio to text using Whisper.
        
        This method performs speech-to-text conversion on the provided
        audio array, returning the transcript along with metadata.
        
        Args:
            audio: NumPy array of audio samples (16kHz, mono, normalized)
        
        Returns:
            Dictionary containing:
                - text: Transcribed text (empty string if silence/noise)
                - language: Detected language code ("ta" for Tamil)
                - confidence: Confidence score (0.0-1.0)
        
        Note:
            - Empty or near-silent audio returns {"text": "", "language": "ta", "confidence": 0.0}
            - All exceptions are caught and re-raised as RuntimeError
        """
        try:
            # Check for empty/silent audio
            if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
                logger.warning("Empty or silent audio detected")
                return {"text": "", "language": "ta", "confidence": 0.0}
            
            # Perform transcription with Tamil language constraint
            result = self._pipeline(
                audio,
                language="ta",  # Tamil
                task="transcribe",
                return_timestamps=False
            )
            
            # Extract text and handle potential None
            text = result.get("text", "")
            
            # Handle empty transcription
            if text is None or not text.strip():
                logger.info("No speech detected in audio")
                return {"text": "", "language": "ta", "confidence": 0.0}
            
            # Extract confidence if available
            chunks = result.get("chunks", [])
            if chunks and len(chunks) > 0:
                # Average confidence across chunks if available
                confidences = [
                    chunk.get("confidence", 1.0)
                    for chunk in chunks
                    if isinstance(chunk, dict) and "confidence" in chunk
                ]
                confidence = sum(confidences) / len(confidences) if confidences else 1.0
            else:
                confidence = 1.0  # Default if no confidence info
            
            output = {
                "text": text.strip(),
                "language": "ta",
                "confidence": float(confidence)
            }
            
            logger.info(
                f"Transcription complete: {len(text)} chars, "
                f"confidence={confidence:.3f}"
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"ASR transcription failed: {e}")
