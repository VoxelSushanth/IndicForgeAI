#!/usr/bin/env python3
"""
Repository Structure Scaffolding Script.

This script automatically creates the missing directories and placeholder
files required for strict monorepo structural compliance. It ensures that
all necessary folders exist with appropriate dummy files if they are missing.

Usage:
    python setup_repo_structure.py

The script is idempotent - running it multiple times will not overwrite
existing files or cause errors.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


# Define the directory structure relative to the project root
# Each entry is a tuple of (directory_path, list_of_files)
REPO_STRUCTURE: List[Tuple[str, List[str]]] = [
    # tests directory
    ("tests", ["test_pipeline.py"]),
    
    # reports directory
    ("reports", ["project1_summary.pdf", "project2_summary.pdf"]),
    
    # docs/architecture directory
    ("docs/architecture", ["system_architecture.png", "buffer_flow_diagram.png"]),
    
    # docs/recordings directory
    ("docs/recordings", ["demo_links.md"]),
    
    # presentation directory
    ("presentation", ["final_presentation.pptx", "evaluation_video_link.md"]),
]


def get_project_root() -> Path:
    """
    Determine the project root directory.
    
    This function looks for the task2_asr_transliteration directory
    and returns its parent as the project root.
    
    Returns:
        Path object pointing to the project root directory
    
    Raises:
        RuntimeError: If project root cannot be determined
    """
    current_dir = Path.cwd()
    
    # Check if we're already in the project root
    if (current_dir / "task2_asr_transliteration").exists():
        return current_dir
    
    # Check if we're inside the app directory
    if current_dir.name == "task2_asr_transliteration":
        return current_dir.parent
    
    # Default to current directory
    print(f"Warning: Could not determine project root, using current directory: {current_dir}")
    return current_dir


def create_directory(directory_path: Path) -> bool:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path object pointing to the directory to create
    
    Returns:
        True if directory was created, False if it already existed
    """
    try:
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory_path}")
            return True
        else:
            print(f"  Directory exists: {directory_path}")
            return False
    except Exception as e:
        print(f"✗ Failed to create directory {directory_path}: {e}")
        return False


def create_dummy_file(file_path: Path, file_extension: str) -> bool:
    """
    Create an empty dummy file or placeholder with appropriate content.
    
    Args:
        file_path: Path object pointing to the file to create
        file_extension: File extension to determine content type
    
    Returns:
        True if file was created successfully, False otherwise
    """
    try:
        if file_path.exists():
            print(f"  File exists: {file_path}")
            return False
        
        # Create different content based on file type
        if file_extension == ".py":
            content = '''"""
Test Module for ASR Pipeline.

This module contains unit tests for the ASRPipeline class,
including tests for audio preprocessing, transcription,
and error handling.
"""

import unittest
import numpy as np
from pathlib import Path


class TestASRPipeline(unittest.TestCase):
    """Test cases for ASRPipeline class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        pass
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        pass
    
    def test_preprocess_audio(self) -> None:
        """Test audio preprocessing functionality."""
        # TODO: Implement test
        pass
    
    def test_transcribe(self) -> None:
        """Test transcription functionality."""
        # TODO: Implement test
        pass
    
    def test_empty_audio(self) -> None:
        """Test handling of empty audio input."""
        # TODO: Implement test
        pass


if __name__ == "__main__":
    unittest.main()
'''
        elif file_extension == ".md":
            if "demo_links" in file_path.name:
                content = '''# Demo Links

This document contains links to demonstration videos and recordings.

## Recorded Demos

- [Demo 1: Basic Transcription](#)
- [Demo 2: Transliteration Features](#)
- [Demo 3: Model Comparison](#)

## Live Demo

- [Gradio Interface](http://localhost:7860)

## Notes

Replace the placeholder links above with actual demo URLs.
'''
            elif "evaluation_video" in file_path.name:
                content = '''# Evaluation Video Link

## Project Evaluation Recording

**Link:** [Insert video link here]

## Presentation Details

- **Duration:** 10-15 minutes
- **Topics Covered:**
  - System Architecture
  - Implementation Details
  - Performance Metrics
  - Future Improvements

## Instructions

1. Upload the evaluation video to your preferred platform
2. Replace this placeholder with the actual video URL
3. Ensure the video is accessible to evaluators
'''
            else:
                content = f"# {file_path.stem}\n\nPlaceholder documentation file.\n"
        
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            # For image files, create a minimal valid PNG file
            # This is a 1x1 transparent PNG
            content = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
                0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,  # bit depth, color type
                0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT chunk
                0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,  # compressed data
                0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,  # 
                0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,  # IEND chunk
                0x42, 0x60, 0x82  # 
            ])
        elif file_extension in [".pdf", ".pptx"]:
            # For binary formats, create a minimal placeholder text file
            # that indicates what should be there
            content = f"""{file_path.name} - Placeholder File

This is a placeholder for the actual {file_path.suffix} file.

For PDF files: Please add the actual project summary document.
For PPTX files: Please add the final presentation slides.

Expected content:
- Project overview
- Architecture diagrams
- Implementation details
- Results and metrics
- Conclusions and future work
"""
            # Write as text but keep the original extension
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Created placeholder file: {file_path}")
            return True
        else:
            content = ""
        
        # Write the file
        if isinstance(content, bytes):
            with open(file_path, 'wb') as f:
                f.write(content)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✓ Created file: {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create file {file_path}: {e}")
        return False


def scaffold_repository_structure() -> Tuple[int, int]:
    """
    Scaffold the complete repository structure.
    
    Creates all missing directories and placeholder files as defined
    in REPO_STRUCTURE.
    
    Returns:
        Tuple of (directories_created, files_created)
    """
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    print("=" * 60)
    
    dirs_created = 0
    files_created = 0
    
    for dir_rel_path, files in REPO_STRUCTURE:
        # Construct full directory path
        dir_path = project_root / dir_rel_path
        
        # Create directory
        if create_directory(dir_path):
            dirs_created += 1
        
        # Create each file in the directory
        for filename in files:
            file_path = dir_path / filename
            file_extension = file_path.suffix.lower()
            
            if create_dummy_file(file_path, file_extension):
                files_created += 1
    
    return dirs_created, files_created


def main() -> int:
    """
    Main entry point for the scaffolding script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("Repository Structure Scaffolding")
    print("=" * 60)
    print()
    
    try:
        dirs_created, files_created = scaffold_repository_structure()
        
        print()
        print("=" * 60)
        print(f"Scaffolding Complete!")
        print(f"  Directories created: {dirs_created}")
        print(f"  Files created: {files_created}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during scaffolding: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
