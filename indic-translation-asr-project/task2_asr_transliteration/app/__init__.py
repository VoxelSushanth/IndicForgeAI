"""
Tamil ASR Transcription & Transliteration Application Package.

This package provides components for automatic speech recognition
of Tamil audio and transliteration to various romanization schemes.

Modules:
    buffer_manager: Thread-safe audio chunk buffering
    asr_pipeline: Whisper-based speech-to-text pipeline
    transliteration: Multi-scheme script conversion engine
    interface: Gradio web interface builder
    main: Application entry point
"""

from .buffer_manager import AudioBufferManager
from .asr_pipeline import ASRPipeline
from .transliteration import TransliterationEngine

__all__ = [
    "AudioBufferManager",
    "ASRPipeline",
    "TransliterationEngine",
]

__version__ = "1.0.0"
