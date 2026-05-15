"""
Main Entry Point for Tamil ASR Transcription & Transliteration System.

This module serves as the application entry point, initializing all
components and launching the Gradio web interface for user interaction.
"""

import logging
import sys
from pathlib import Path

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Initialize components and launch the Gradio interface.
    
    This function performs the following steps:
    1. Logs application startup
    2. Imports application components with error handling
    3. Builds the Gradio interface
    4. Launches the web server on port 7860
    
    Raises:
        SystemExit: If critical dependencies are missing
    """
    logger.info("=" * 60)
    logger.info("Tamil ASR Transcription & Transliteration System")
    logger.info("=" * 60)
    
    try:
        # Import application components
        from .buffer_manager import AudioBufferManager
        from .asr_pipeline import ASRPipeline
        from .transliteration import TransliterationEngine
        from .interface import build_interface
        
        logger.info("All modules imported successfully")
        
        # Test component initialization (without loading heavy models)
        buffer = AudioBufferManager(maxsize=10)
        logger.info(f"AudioBufferManager initialized: maxsize={buffer.size()}")
        
        transliterator = TransliterationEngine(
            source_scheme="Tamil",
            target_scheme="ITRANS"
        )
        logger.info(
            f"TransliterationEngine initialized: "
            f"{transliterator.source_scheme} -> {transliterator.target_scheme}"
        )
        
        # Build the Gradio interface
        logger.info("Building Gradio interface...")
        demo = build_interface()
        
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
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
