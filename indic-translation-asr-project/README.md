# 🎙️ Indic Translation & ASR Project

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-orange.svg)

## Overview

This project provides a comprehensive suite of tools for **English-to-Tamil translation evaluation** and **Tamil speech recognition with transliteration**. It consists of two main components: (1) A detailed evaluation framework for assessing neural machine translation models on Tamil language pairs using sacreBLEU, chrF, and TER metrics, and (2) A production-ready ASR system leveraging OpenAI Whisper for Tamil transcription with multi-scheme transliteration output via an interactive Gradio web interface.

## Repository Structure

```
indic-translation-asr-project/
├── README.md
├── task1_translation_evaluation/
│   ├── translation_dataset.csv
│   ├── part_a_batch_translation/
│   │   └── part_a_translation_evaluation.ipynb
│   ├── part_b_token_analysis/
│   │   └── part_b_token_eda.ipynb
│   └── part_c_indic_token_behavior/
│       └── part_c_indic_token_analysis.ipynb
└── task2_asr_transliteration/
    ├── app/
    │   ├── __init__.py
    │   ├── buffer_manager.py
    │   ├── asr_pipeline.py
    │   ├── transliteration.py
    │   ├── interface.py
    │   └── main.py
    ├── sample_inputs/
    ├── outputs/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning)

### Installation

```bash
# Clone the repository
cd indic-translation-asr-project

# Install dependencies for Task 2
cd task2_asr_transliteration
pip install -r requirements.txt
```

---

## Task 1: Translation Model Evaluation

### Running the Notebooks

#### Part A: Batch Translation Evaluation

```bash
cd task1_translation_evaluation/part_a_batch_translation
jupyter notebook part_a_translation_evaluation.ipynb
```

This notebook:
- Loads the AI4Bharat IndicTrans2 model
- Performs batch translation on the test dataset
- Computes sacreBLEU, chrF, and TER metrics
- Generates visualization of results

#### Part B: Token-Level EDA

```bash
cd task1_translation_evaluation/part_b_token_analysis
jupyter notebook part_b_token_eda.ipynb
```

This notebook:
- Analyzes tokenization across 5 models
- Computes expansion ratios and fragmentation metrics
- Generates comparative visualizations

#### Part C: Indic Token Behavior Analysis

```bash
cd task1_translation_evaluation/part_c_indic_token_behavior
jupyter notebook part_c_indic_token_analysis.ipynb
```

This notebook:
- Deep analysis of Tamil-specific tokenization
- Memory footprint estimation
- Agglutinative word handling comparison

---

## Task 2: ASR Transcription & Transliteration

### Option 1: Docker Deployment (Recommended)

```bash
cd task2_asr_transliteration

# Build and run with Docker Compose
docker-compose up --build

# Access the interface at http://localhost:7860
```

### Option 2: Local Run

```bash
cd task2_asr_transliteration

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m app.main

# Access the interface at http://localhost:7860
```

### Option 3: Docker Direct

```bash
cd task2_asr_transliteration

# Build the image
docker build -t tamil-asr .

# Run the container
docker run -p 7860:7860 -v $(pwd)/outputs:/app/outputs tamil-asr
```

---

## Results Summary

### Translation Evaluation (Task 1)

| Model | BLEU (char) | chrF | TER | Notes |
|-------|-------------|------|-----|-------|
| ai4bharat/indictrans2-en-indic-1B | *TBD* | *TBD* | *TBD* | Primary model |
| facebook/nllb-200-distilled-600M | *TBD* | *TBD* | *TBD* | Fallback option |
| Helsinki-NLP/opus-mt-en-ta | *TBD* | *TBD* | *TBD* | Baseline |

*Run the notebooks to compute actual scores*

### ASR System Features (Task 2)

| Feature | Description |
|---------|-------------|
| Models | Whisper small/medium |
| Audio Formats | WAV, MP3, OGG, FLAC, M4A, WebM |
| Transliteration | ITRANS, HK, IAST, Devanagari |
| Interface | Gradio web UI |
| Deployment | Docker-ready |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tamil ASR System Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Audio      │    │   Whisper    │    │     Tamil    │      │
│  │   Input      │───▶│   ASR Model  │───▶│ Transcription│      │
│  │ (Upload/Mic) │    │  (16kHz)     │    │   (Tamil)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                  │              │
│                                                  ▼              │
│                                         ┌──────────────┐       │
│                                         │ Transliteration│      │
│                                         │   Engine     │       │
│                                         │ (ITRANS/HK/  │       │
│                                         │  IAST/etc.)  │       │
│                                         └──────────────┘       │
│                                                  │              │
│                                                  ▼              │
│                                         ┌──────────────┐       │
│                                         │   Gradio     │       │
│                                         │   Web UI     │       │
│                                         └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### Task 1 (Translation Evaluation)

```
transformers>=4.35.0
sentencepiece>=0.1.99
sacrebleu>=2.3.0
pandas>=2.0.0
torch>=2.0.0
matplotlib>=3.7.0
```

### Task 2 (ASR System)

See `task2_asr_transliteration/requirements.txt` for complete list.

Key dependencies:
- `openai-whisper` - ASR model
- `gradio` - Web interface
- `librosa` - Audio processing
- `indic-transliteration` - Script conversion

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{indic_translation_asr_2024,
  title = {Indic Translation and ASR Project},
  author = {AIML Engineering Assessment},
  year = {2024},
  url = {https://github.com/your-repo/indic-translation-asr-project}
}
```
