# System Architecture Diagram

This document contains the Mermaid.js flowchart diagram illustrating the complete audio processing pipeline from input to output.

## Audio Processing Pipeline Flowchart

```mermaid
flowchart TD
    subgraph UserInterface["Gradio Web Interface"]
        A[Audio Upload / Microphone Input]
        B[Model Selection Dropdown]
        C[Scheme Selection Dropdown]
        D[Transcript Output TextBox]
        E[Transliteration Output TextBox]
        F[Metadata JSON Display]
    end

    subgraph AsyncProducer["Async Producer (Gradio Handler)"]
        G[process_audio async function]
        H[Preprocess Audio Array]
        I[Generate Job ID]
        J[Enqueue to Buffer Manager]
    end

    subgraph BufferManager["Thread-Safe Queue Buffer"]
        K[Queue: audio_array, job_id, target_scheme]
        L[Max Size: 10 jobs]
    end

    subgraph BackgroundWorker["Consumer Worker Thread"]
        M[Worker Loop - queue.get]
        N[ASR Pipeline - Whisper]
        O[Transliteration Engine]
        P[Update async_responses dict]
    end

    subgraph ModelCache["Global Model Cache"]
        Q[Whisper small/medium pipelines]
        R[Transliterator with IndicNLP]
    end

    subgraph AsyncResponses["Thread-Safe Response Dictionary"]
        S[job_id: status, result, event]
    end

    %% Flow connections
    A --> G
    B --> G
    C --> G
    
    G --> H
    H --> I
    I --> J
    J --> K
    
    K --> M
    M --> N
    M --> O
    
    N -.-> Q
    O -.-> R
    
    N --> P
    O --> P
    P --> S
    
    S -.-> G
    
    G --> D
    G --> E
    G --> F

    %% Styling
    style UserInterface fill:#e1f5fe
    style AsyncProducer fill:#fff3e0
    style BufferManager fill:#f3e5f5
    style BackgroundWorker fill:#e8f5e9
    style ModelCache fill:#ffebee
    style AsyncResponses fill:#fff8e1
```

## Component Descriptions

| Component | Responsibility |
|-----------|----------------|
| **Gradio Web Interface** | Handles user interactions, audio upload/recording, displays results |
| **Async Producer** | Preprocesses audio, generates job IDs, enqueues work items |
| **Buffer Manager** | Thread-safe queue with max size limit for backpressure |
| **Background Worker** | Consumes jobs, runs Whisper ASR and transliteration |
| **Model Cache** | Global singleton cache for loaded ML models |
| **Async Responses** | Thread-safe dictionary for job status/result communication |

## Data Flow

1. User uploads/records audio via Gradio interface
2. `process_audio` async function preprocesses audio to numpy array
3. Unique job ID generated and job registered in `async_responses`
4. Audio tuple enqueued to buffer manager (producer)
5. Background worker dequeues job (consumer)
6. Whisper pipeline transcribes audio to Tamil text
7. Transliteration engine converts to target script scheme
8. Results stored in `async_responses` dictionary
9. Async polling loop detects completion and streams results to UI
