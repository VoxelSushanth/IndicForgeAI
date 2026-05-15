"""
Audio Buffer Manager for ASR Pipeline.

This module provides a thread-safe queue-based buffer manager for handling
audio chunks in the ASR transcription pipeline. It ensures safe concurrent
access from producer and consumer threads without raising exceptions.
"""

import logging
import queue
from typing import List, Optional
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class AudioBufferManager:
    """
    Thread-safe audio buffer manager using a bounded queue.
    
    This class manages audio chunks in a FIFO queue with configurable
    maximum size. All methods are thread-safe and handle edge cases
    gracefully without raising exceptions to callers.
    
    Attributes:
        _queue: Internal queue.Queue instance with maxsize=10
        _maxsize: Maximum number of items the queue can hold
    
    Example:
        >>> manager = AudioBufferManager()
        >>> manager.enqueue(audio_chunk)
        True
        >>> chunk = manager.dequeue()
        array([...])
    """
    
    def __init__(self, maxsize: int = 10) -> None:
        """
        Initialize the audio buffer manager.
        
        Args:
            maxsize: Maximum number of audio chunks to buffer (default: 10)
        """
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._maxsize: int = maxsize
        logger.info(f"AudioBufferManager initialized with maxsize={maxsize}")
    
    def enqueue(self, audio_chunk: np.ndarray, timeout: float = 5.0) -> bool:
        """
        Add an audio chunk to the buffer queue.
        
        This method attempts to add the provided audio chunk to the internal
        queue. If the queue is full, it logs a warning and returns False
        without raising an exception.
        
        Args:
            audio_chunk: NumPy array containing audio samples
            timeout: Maximum time to wait for space in queue (seconds)
        
        Returns:
            True if chunk was successfully enqueued, False if queue was full
        
        Note:
            This method is thread-safe and will not raise exceptions.
            Failed enqueue operations are logged at WARNING level.
        """
        try:
            self._queue.put(audio_chunk, block=True, timeout=timeout)
            logger.debug(f"Enqueued audio chunk of shape {audio_chunk.shape}")
            return True
        except queue.Full:
            logger.warning(
                f"Queue is full (maxsize={self._maxsize}), "
                f"dropping audio chunk of shape {audio_chunk.shape}"
            )
            return False
        except Exception as e:
            logger.error(f"Unexpected error during enqueue: {e}")
            return False
    
    def dequeue(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        """
        Remove and return an audio chunk from the buffer queue.
        
        This method retrieves the oldest audio chunk from the queue.
        If the queue is empty, it logs a debug message and returns None
        without raising an exception.
        
        Args:
            timeout: Maximum time to wait for an item (seconds)
        
        Returns:
            NumPy array containing audio samples, or None if queue was empty
        
        Note:
            This method is thread-safe and will not raise exceptions.
            Empty queue conditions are handled gracefully.
        """
        try:
            audio_chunk = self._queue.get(block=True, timeout=timeout)
            logger.debug(f"Dequeued audio chunk of shape {audio_chunk.shape}")
            return audio_chunk
        except queue.Empty:
            logger.debug("Queue is empty, no audio chunk available")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during dequeue: {e}")
            return None
    
    def flush(self) -> List[np.ndarray]:
        """
        Drain all items from the queue and return them as a list.
        
        This method removes all pending audio chunks from the queue
        and returns them in FIFO order. The queue will be empty after
        this operation completes.
        
        Returns:
            List of NumPy arrays containing all buffered audio chunks
        
        Note:
            This method is thread-safe. The returned list contains references
            to the original arrays, not copies.
        """
        chunks: List[np.ndarray] = []
        
        while True:
            try:
                chunk = self._queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error during flush: {e}")
                break
        
        logger.info(f"Flushed {len(chunks)} audio chunks from buffer")
        return chunks
    
    def is_empty(self) -> bool:
        """
        Check if the buffer queue is empty.
        
        Returns:
            True if queue contains no items, False otherwise
        
        Note:
            Due to concurrent access, the result may become stale
            immediately after returning. Use for informational purposes.
        """
        return self._queue.empty()
    
    def size(self) -> int:
        """
        Get the current number of items in the buffer queue.
        
        Returns:
            Number of audio chunks currently in the queue
        
        Note:
            Due to concurrent access, the result may become stale
            immediately after returning. Use for informational purposes.
        """
        return self._queue.qsize()
