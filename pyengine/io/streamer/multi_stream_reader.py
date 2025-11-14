import threading
import time
from collections import deque
from typing import Dict, List, Optional, Any

from pyengine.io.streamer.stream_reader import StreamReader
from pyengine.utils.logger import logger


class MultiVideoReader:
    """
    Multi-source video reader with polling and buffering capabilities.

    Features:
    - Poll multiple video sources simultaneously
    - Per-source buffering with configurable size
    - Flexible batch reading: by name, one per source, or total count

    Usage:
        # Initialize with multiple sources
        reader = MultiVideoReader(max_buffer_per_source=100)
        reader.add_source("cam1", "rtsp://...", width=640, height=480)
        reader.add_source("cam2", "0", width=640, height=480)
        reader.start()

        # Read by source name
        frame_info = reader.get_by_name("cam1")

        # Read one frame from each source
        batch = reader.get_one_per_source()

        # Read N frames total (mixed sources)
        frames = reader.get_batch(10)

        # Stop reading
        reader.stop()
    """

    def __init__(self, max_buffer_per_source: int = 1000):
        """
        Initialize MultiVideoReader.

        Args:
            max_buffer_per_source: Maximum buffer size per source
        """
        self.max_buffer_per_source = max_buffer_per_source

        # Source configuration: {name: {reader, thread, buffer, lock}}
        self._sources: Dict[str, Dict[str, Any]] = {}

        # Global control
        self._stop_event = threading.Event()
        self._running = False

        logger.info("MultiVideoReader", f"Initialized with buffer size {max_buffer_per_source} per source")

    def add_source(self, name: str, url: str,
                   width: int = -1, height: int = -1, fps: int = -1,
                   max_retries: int = 5, retry_delay: int = 2, loop_video: bool = True) -> 'MultiVideoReader':
        """
        Add a video source.

        Args:
            name: Unique identifier for this source
            url: Video source (RTSP URL, camera index, or file path)
            width: Target width (-1 for native)
            height: Target height (-1 for native)
            fps: Target FPS (-1 for native/unlimited)
            max_retries: Maximum reconnection attempts
            retry_delay: Delay between reconnection attempts
            loop_video: Whether to loop video files when they reach the end (default: True)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source name already exists
        """
        if name in self._sources:
            raise ValueError(f"Source '{name}' already exists")

        if self._running:
            raise RuntimeError("Cannot add sources while reader is running. Call stop() first.")

        # Create StreamReader
        reader = StreamReader(
            url=url,
            width=width,
            height=height,
            fps=fps,
            max_retries=max_retries,
            retry_delay=retry_delay,
            loop_video=loop_video
        )

        # Create source entry
        self._sources[name] = {
            'reader': reader,
            'thread': None,
            'buffer': deque(maxlen=self.max_buffer_per_source),
            'lock': threading.Lock(),  # For buffer operations
            'reader_lock': threading.Lock(),  # For reader connection operations
            'last_ts': 0.0,
            'url': url,
            'opened': False
        }

        logger.info("MultiVideoReader", f"Added source '{name}' -> {url}")
        return self

    def remove_source(self, name: str) -> bool:
        """
        Remove a video source.

        Args:
            name: Source identifier to remove

        Returns:
            True if removed, False if not found
        """
        if self._running:
            raise RuntimeError("Cannot remove sources while reader is running. Call stop() first.")

        if name not in self._sources:
            logger.warning("MultiVideoReader", f"Source '{name}' not found")
            return False

        # Clean up
        source = self._sources[name]
        if source['opened']:
            source['reader'].close()

        del self._sources[name]
        logger.info("MultiVideoReader", f"Removed source '{name}'")
        return True

    def start(self):
        """Start polling all sources."""
        if self._running:
            logger.warning("MultiVideoReader", "Already running")
            return

        if not self._sources:
            raise RuntimeError("No sources added. Use add_source() first.")

        self._stop_event.clear()
        self._running = True

        # Track successfully started sources for rollback on failure
        started_sources = []

        try:
            # Start a polling thread for each source
            for name, source in self._sources.items():
                try:
                    source['reader'].open()
                    source['opened'] = True

                    thread = threading.Thread(
                        target=self._poll_source,
                        args=(name,),
                        name=f"Poll-{name}",
                        daemon=True
                    )
                    thread.start()
                    source['thread'] = thread
                    started_sources.append(name)

                    logger.info("MultiVideoReader", f"Started polling source '{name}'")
                except Exception as e:
                    logger.error("MultiVideoReader", f"Failed to start source '{name}': {e}")
                    # Rollback: stop all previously started sources
                    self._rollback_startup(started_sources)
                    self._running = False
                    raise RuntimeError(f"Failed to start all sources, rolled back. Error: {e}")

            logger.info("MultiVideoReader", f"Started polling {len(started_sources)} sources")
        except Exception as e:
            # Ensure running flag is cleared on any error
            self._running = False
            raise

    def _rollback_startup(self, started_sources: List[str]):
        """
        Rollback startup by stopping all sources that were successfully started.

        Args:
            started_sources: List of source names that were successfully started
        """
        logger.warning("MultiVideoReader", f"Rolling back startup for {len(started_sources)} sources")
        self._stop_event.set()

        for name in started_sources:
            source = self._sources.get(name)
            if not source:
                continue

            # Wait for thread to finish with timeout protection
            thread = source['thread']
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

                # Check if thread actually stopped
                if thread.is_alive():
                    logger.warning("MultiVideoReader",
                                 f"Thread for source '{name}' did not stop after 2s timeout during rollback. "
                                 f"Thread may be hung. State: {thread.is_alive()}")
                else:
                    logger.debug("MultiVideoReader", f"Thread for source '{name}' stopped successfully during rollback")

            # Close reader
            if source['opened']:
                try:
                    source['reader'].close()
                    source['opened'] = False
                except Exception as e:
                    logger.error("MultiVideoReader", f"Error closing source '{name}' during rollback: {e}")

        # Clear the stop event for potential retry
        self._stop_event.clear()

    def stop(self):
        """Stop polling all sources."""
        if not self._running:
            return

        logger.info("MultiVideoReader", "Stopping all sources...")
        self._stop_event.set()
        self._running = False

        # Wait for all threads to finish with timeout protection
        hung_threads = []
        for name, source in self._sources.items():
            thread = source['thread']
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

                # Check if thread actually stopped
                if thread.is_alive():
                    hung_threads.append(name)
                    logger.warning("MultiVideoReader",
                                 f"Thread for source '{name}' did not stop after 2s timeout. "
                                 f"Thread may be hung or blocked on I/O.")
                else:
                    logger.debug("MultiVideoReader", f"Thread for source '{name}' stopped successfully")

            # Close reader (attempt even if thread is hung)
            if source['opened']:
                try:
                    source['reader'].close()
                    source['opened'] = False
                    logger.debug("MultiVideoReader", f"Reader for source '{name}' closed successfully")
                except Exception as e:
                    logger.error("MultiVideoReader", f"Error closing source '{name}': {e}")

        # Report hung threads summary
        if hung_threads:
            logger.error("MultiVideoReader",
                        f"Warning: {len(hung_threads)} thread(s) did not stop cleanly: {hung_threads}. "
                        f"These threads may continue running in the background.")

        logger.info("MultiVideoReader", "All sources stopped")

    def _poll_source(self, name: str):
        """
        Polling thread for a single source.

        Args:
            name: Source identifier
        """
        source = self._sources[name]
        reader = source['reader']
        buffer = source['buffer']
        lock = source['lock']
        reader_lock = source['reader_lock']

        consecutive_failures = 0
        max_consecutive_failures = 50

        while not self._stop_event.is_set():
            try:
                # Check connection and reconnect atomically
                # Use reader_lock to prevent race conditions between is_connected() and reconnect()
                with reader_lock:
                    if not reader.is_connected():
                        logger.warning("MultiVideoReader", f"Source '{name}' disconnected, attempting reconnect...")
                        if reader.reconnect():
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                logger.error("MultiVideoReader", f"Source '{name}' failed too many times, stopping polling")
                                break
                            time.sleep(1.0)
                            continue

                # Read frame (outside reader_lock to avoid blocking)
                frame = reader.read()

                if frame is not None:
                    consecutive_failures = 0
                    ts = time.time()

                    # Add to buffer (use separate lock for buffer operations)
                    with lock:
                        buffer.append({
                            'name': name,
                            'frame': frame,
                            'last_ts': ts
                        })
                        source['last_ts'] = ts
                else:
                    # No frame ready (FPS throttling), short sleep
                    time.sleep(0.001)

            except Exception as e:
                logger.error("MultiVideoReader", f"Error polling source '{name}': {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("MultiVideoReader", f"Source '{name}' failed too many times, stopping polling")
                    break
                time.sleep(0.1)

        logger.info("MultiVideoReader", f"Polling thread for '{name}' exited")

    def get_by_name(self, name: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Get the latest frame from a specific source.

        Args:
            name: Source identifier
            timeout: Maximum wait time if buffer is empty (0 = no wait)

        Returns:
            Dict with keys: 'name', 'frame', 'last_ts', or None if not available
        """
        if name not in self._sources:
            logger.warning("MultiVideoReader", f"Source '{name}' not found")
            return None

        source = self._sources[name]
        buffer = source['buffer']
        lock = source['lock']

        deadline = time.time() + timeout

        while True:
            with lock:
                if buffer:
                    return buffer.popleft()

            if timeout <= 0 or time.time() >= deadline:
                return None

            time.sleep(0.001)

    def get_one_per_source(self, timeout: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get one frame from each source (if available).

        Args:
            timeout: Maximum wait time per source if buffer is empty

        Returns:
            List of frame info dicts (may be less than total sources if some are empty)
        """
        results = []

        for name in self._sources.keys():
            frame_info = self.get_by_name(name, timeout=timeout)
            if frame_info is not None:
                results.append(frame_info)

        return results

    def get_one_per_source_balanced(self, timeout: float = 0.0, max_wait_iterations: int = 10) -> List[Dict[str, Any]]:
        """
        Get one frame from each source with balanced round-robin polling.

        This method ensures fair treatment of all video sources by using a round-robin
        approach to poll sources, preventing fast sources from starving slow sources.

        Key features:
        - Round-robin polling across all sources
        - Tracks which sources have provided frames
        - Continues polling until all sources provide a frame or timeout/max iterations reached
        - Prevents starvation of slow video sources

        Args:
            timeout: Maximum total wait time for all sources (seconds)
            max_wait_iterations: Maximum number of polling iterations before giving up

        Returns:
            List of frame info dicts (one per source if available within timeout)

        Example:
            # Wait up to 0.1 seconds for all sources to provide frames
            batch = reader.get_one_per_source_balanced(timeout=0.1, max_wait_iterations=20)
        """
        if not self._sources:
            return []

        source_names = list(self._sources.keys())
        results_dict = {}  # Use dict to track which sources have provided frames
        deadline = time.time() + timeout if timeout > 0 else float('inf')
        iteration = 0
        source_idx = 0

        # Continue until we have frames from all sources or reach timeout/max iterations
        while len(results_dict) < len(source_names) and iteration < max_wait_iterations:
            # Check timeout
            if time.time() >= deadline:
                break

            # Get current source name in round-robin fashion
            name = source_names[source_idx]

            # Only poll this source if we haven't gotten a frame from it yet
            if name not in results_dict:
                source = self._sources[name]

                # Try to get frame from buffer (non-blocking)
                with source['lock']:
                    if source['buffer']:
                        frame_info = source['buffer'].popleft()
                        results_dict[name] = frame_info

            # Move to next source (round-robin)
            source_idx = (source_idx + 1) % len(source_names)

            # If we've completed a full round without getting any new frames, short sleep
            if source_idx == 0:
                iteration += 1
                if len(results_dict) < len(source_names):
                    # Brief sleep to avoid busy-waiting
                    time.sleep(0.001)

        # Convert dict to list, maintaining source order
        return [results_dict[name] for name in source_names if name in results_dict]

    def get_batch(self, count: int, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get N frames total from all sources (round-robin style).

        Args:
            count: Number of frames to retrieve
            timeout: Maximum total wait time

        Returns:
            List of frame info dicts (may be less than count if timeout reached)
        """
        results = []
        deadline = time.time() + timeout
        source_names = list(self._sources.keys())

        if not source_names:
            return results

        source_idx = 0

        while len(results) < count and time.time() < deadline:
            name = source_names[source_idx]
            source = self._sources[name]

            with source['lock']:
                if source['buffer']:
                    results.append(source['buffer'].popleft())

            # Move to next source (round-robin)
            source_idx = (source_idx + 1) % len(source_names)

            # If we've checked all sources and got nothing, short sleep
            if source_idx == 0 and len(results) == 0:
                time.sleep(0.001)

        return results

    def get_buffer_sizes(self) -> Dict[str, int]:
        """
        Get current buffer size for each source.

        Returns:
            Dict mapping source name to buffer size
        """
        sizes = {}
        for name, source in self._sources.items():
            with source['lock']:
                sizes[name] = len(source['buffer'])
        return sizes

    def get_source_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all sources.

        Returns:
            Dict with source info (name, url, buffer size, last timestamp, connected status)
        """
        info = {}
        for name, source in self._sources.items():
            with source['lock']:
                info[name] = {
                    'url': source['url'],
                    'buffer_size': len(source['buffer']),
                    'last_ts': source['last_ts'],
                    'connected': source['reader'].is_connected() if source['opened'] else False
                }
        return info

    def is_running(self) -> bool:
        """Check if the reader is running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        if self._running:
            self.stop()


"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time

from pyengine.io.streamer.multi_video_reader import MultiVideoReader


def example_basic_usage():
    # Basic usage example.
    
    print("=== Basic Usage Example ===")
    
    # Create reader
    reader = MultiVideoReader(max_buffer_per_source=100)
    
    # Add multiple sources
    reader.add_source("cam1", "0", width=640, height=480, fps=30)  # Local camera
    reader.add_source("cam2", "path/to/video.mp4", width=640, height=480, fps=30)  # Video file
    # reader.add_source("cam3", "rtsp://user:pass@192.168.1.10/stream", width=640, height=480)  # RTSP stream
    
    # Start polling
    reader.start()
    
    try:
        frame_count = 0
        while frame_count < 100:
            # Get one frame from each source
            batch = reader.get_one_per_source(timeout=0.1)
            
            for frame_info in batch:
                name = frame_info['name']
                frame = frame_info['frame']
                ts = frame_info['last_ts']
                
                # Display
                cv2.imshow(f"Source: {name}", frame)
                print(f"Frame from {name} at {ts:.3f}")
                frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        reader.stop()
        cv2.destroyAllWindows()


def example_get_by_name():
    # Example: Reading from specific source.
    print("\n=== Get By Name Example ===")
    
    reader = MultiVideoReader(max_buffer_per_source=50)
    reader.add_source("primary", "0", width=640, height=480)
    reader.start()
    
    try:
        for _ in range(50):
            # Get frame from specific source
            frame_info = reader.get_by_name("primary", timeout=0.1)
            
            if frame_info:
                print(f"Got frame from {frame_info['name']} at {frame_info['last_ts']:.3f}")
                cv2.imshow("Primary Camera", frame_info['frame'])
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    
    finally:
        reader.stop()
        cv2.destroyAllWindows()


def example_batch_reading():
    # Example: Batch reading mixed sources.
    
    print("\n=== Batch Reading Example ===")
    
    reader = MultiVideoReader(max_buffer_per_source=100)
    reader.add_source("src1", "0", width=320, height=240)
    reader.add_source("src2", "video.mp4", width=320, height=240)
    reader.start()
    
    try:
        for i in range(10):
            # Get 10 frames total (mixed from all sources)
            batch = reader.get_batch(count=10, timeout=0.5)
            
            print(f"\nBatch {i}: Got {len(batch)} frames")
            for frame_info in batch:
                print(f"  - {frame_info['name']} @ {frame_info['last_ts']:.3f}")
            
            # Check buffer status
            buffer_sizes = reader.get_buffer_sizes()
            print(f"Buffer sizes: {buffer_sizes}")
            
            time.sleep(0.1)
    
    finally:
        reader.stop()


def example_context_manager():
    # Example: Using context manager.
    print("\n=== Context Manager Example ===")
    
    with MultiVideoReader(max_buffer_per_source=50) as reader:
        reader.add_source("cam", "0", width=640, height=480)
        # start() is called automatically
        
        for _ in range(30):
            batch = reader.get_one_per_source(timeout=0.1)
            for frame_info in batch:
                cv2.imshow("Camera", frame_info['frame'])
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    
    # stop() is called automatically
    cv2.destroyAllWindows()


def example_monitoring():
    # Example: Monitoring source status.
    print("\n=== Monitoring Example ===")
    
    reader = MultiVideoReader(max_buffer_per_source=100)
    reader.add_source("cam1", "0")
    reader.add_source("cam2", "video.mp4")
    reader.start()
    
    try:
        for i in range(20):
            # Get source information
            info = reader.get_source_info()
            
            print(f"\n--- Status Update {i} ---")
            for name, details in info.items():
                print(f"{name}:")
                print(f"  URL: {details['url']}")
                print(f"  Connected: {details['connected']}")
                print(f"  Buffer: {details['buffer_size']} frames")
                print(f"  Last frame: {details['last_ts']:.3f}")
            
            time.sleep(1.0)
    
    finally:
        reader.stop()


if __name__ == "__main__":
    # Run examples
    print("MultiVideoReader Examples\n")
    
    # Choose which example to run
    example_basic_usage()
    # example_get_by_name()
    # example_batch_reading()
    # example_context_manager()
    # example_monitoring()
"""