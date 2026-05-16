# Buffer Flow Diagram

This document contains the Mermaid.js sequence diagram illustrating the async producer-consumer threading model between Gradio, the Buffer Manager, and the Worker Thread.

## Async Producer-Consumer Sequence Diagram

```mermaid
sequenceDiagram
    participant UI as Gradio UI
    participant Handler as process_audio (Async)
    participant Cache as Model Cache
    participant Queue as Buffer Manager Queue
    participant Worker as Background Worker Thread
    participant ASR as Whisper Pipeline
    participant Translit as Transliteration Engine
    participant Responses as async_responses Dict

    Note over UI,Responses: Phase 1: Audio Submission (Producer)
    
    UI->>Handler: User clicks "Transcribe & Transliterate"
    activate Handler
    
    Handler->>Cache: Get ASR pipeline for model_size
    Cache-->>Handler: Return ASRPipeline instance
    
    Handler->>Handler: Preprocess audio to numpy array
    
    Handler->>Handler: Generate unique job_id
    
    Handler->>Responses: Register job: {status: pending}
    activate Responses
    
    Handler->>Queue: Enqueue (audio_array, job_id, scheme)
    activate Queue
    
    Queue-->>Handler: Enqueue success
    
    Handler->>UI: Yield "Queued..." status
    
    Handler->>UI: Yield "Processing Audio..." status
    
    Note over UI,Responses: Phase 2: Async Polling Loop
    
    loop Poll every 1 second (non-blocking)
        Handler->>Responses: Check job_id status
        Responses-->>Handler: Return status (pending/processing)
        Handler->>UI: Yield "Processing... (Xs elapsed)"
        Handler->>Handler: await asyncio.sleep(1)
    end
    
    Note over UI,Responses: Phase 3: Worker Processing (Consumer)
    
    Queue->>Worker: queue.get() blocks until item available
    deactivate Queue
    
    Worker->>Worker: Unpack (audio_array, job_id, scheme)
    
    Worker->>ASR: transcribe(audio_array)
    activate ASR
    ASR-->>Worker: {text: Tamil transcript, confidence}
    deactivate ASR
    
    Worker->>Translit: transliterate(text, scheme)
    activate Translit
    Translit-->>Worker: Romanized/Devanagari text
    deactivate Translit
    
    Worker->>Responses: Update job_id: {status: completed, result}
    
    Note over UI,Responses: Phase 4: Result Delivery
    
    Handler->>Responses: Check job_id status
    Responses-->>Handler: status = "completed"
    
    Handler->>Responses: Retrieve result and cleanup
    deactivate Responses
    
    Handler->>UI: Yield (transcript, transliteration, metadata)
    deactivate Handler
    
    Note over UI,Responses: Alternative Path: Error Handling
    
    alt Error during processing
        Worker->>Responses: Update job_id: {status: error, error_msg}
        Handler->>Responses: Detect error status
        Handler->>UI: Yield "Error: error_msg"
    end
    
    alt Timeout after 120s
        Handler->>Handler: elapsed_time >= max_wait_time
        Handler->>UI: Yield "Error: Processing timed out"
    end
```

## Key Design Patterns

### 1. Producer-Consumer Pattern
- **Producer**: Gradio's `process_audio` async function enqueues work items
- **Consumer**: Dedicated background worker thread dequeues and processes
- **Benefit**: Decouples UI responsiveness from heavy ML inference

### 2. Async Polling with Yield
- **Traditional approach**: `job_event.wait(timeout=120.0)` - blocks thread
- **New approach**: `while True: await asyncio.sleep(1)` - non-blocking
- **Benefit**: Frees Gradio/FastAPI worker thread for other requests

### 3. Live Status Streaming
- Uses Gradio's generator/yield support for real-time UI updates
- User sees: "Queued..." → "Processing Audio..." → "Processing... (5s)" → Results
- **Benefit**: Eliminates perceived freezing, improves UX

### 4. Thread-Safe Communication
- `async_responses` dictionary protected by `threading.Lock`
- Job registration, status updates, and result retrieval are atomic
- **Benefit**: Prevents race conditions between threads

## Timing Characteristics

| Operation | Typical Duration | Blocking? |
|-----------|------------------|-----------|
| Audio preprocessing | < 100ms | No (fast numpy ops) |
| Queue enqueue | < 10ms | No (in-memory) |
| asyncio.sleep(1) | 1 second | No (yields to event loop) |
| Whisper inference | 2-30s | Yes (in worker thread) |
| Transliteration | < 1s | Yes (in worker thread) |
| Response polling | < 1ms | No (dict lookup) |

## Concurrency Guarantees

1. **Queue Thread-Safety**: `queue.Queue` provides internal locking
2. **Response Dict Safety**: Explicit `threading.Lock` for all access
3. **Job Isolation**: Unique job_id prevents cross-talk between requests
4. **Backpressure**: Queue max size (10) prevents memory exhaustion
