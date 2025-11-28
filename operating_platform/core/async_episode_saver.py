#!/usr/bin/env python

"""
Asynchronous Episode Saver for DoRobot Dataset

This module provides asynchronous episode saving functionality to eliminate
blocking delays during data collection. It allows immediate return to the
recording loop while save operations (image writing, video encoding, metadata
updates) happen in background threads.

Key features:
- Immediate metadata return (<100ms)
- Background queue processing
- Thread-safe episode index allocation
- Retry logic for failed saves
- Status monitoring and error tracking
"""

import copy
import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeSaveTask:
    """
    Represents a single episode save operation to be processed asynchronously.

    Attributes:
        episode_index: The unique index for this episode
        episode_buffer: Deep copy of episode data (isolated from main thread)
        dataset: Reference to DoRobotDataset instance
        record_cfg: Recording configuration
        record_cmd: Recording command metadata
        timestamp: When this task was created
        retry_count: Number of retry attempts made
        max_retries: Maximum retry attempts before failure
    """
    episode_index: int
    episode_buffer: dict
    dataset: 'DoRobotDataset'  # Forward reference
    record_cfg: 'RecordConfig'  # Forward reference
    record_cmd: dict
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class EpisodeMetadata:
    """
    Immediate return value when save is requested.

    This is returned instantly to the caller, allowing them to continue
    without waiting for the actual save to complete.

    Attributes:
        episode_index: The assigned episode index
        last_record_episode_index: Same as episode_index (for compatibility)
        estimated_frames: Number of frames in this episode
        task_queued: Whether task was successfully queued
        queue_position: Position in the save queue (0-indexed)
    """
    episode_index: int
    last_record_episode_index: int
    estimated_frames: int
    task_queued: bool
    queue_position: int


class AsyncEpisodeSaver:
    """
    Handles asynchronous episode saving with background worker thread.

    This class provides immediate metadata return while actual save operations
    (image writing, video encoding, metadata updates) happen in the background.

    The design ensures:
    - Data format compatibility (uses existing save_episode() method)
    - Thread safety (locks for shared state)
    - Error resilience (retry logic)
    - Progress monitoring (status tracking)

    Usage:
        saver = AsyncEpisodeSaver(max_queue_size=10)
        saver.start(initial_episode_index=0)

        # Quick return, no blocking
        metadata = saver.queue_save(episode_buffer, dataset, cfg, cmd)

        # Before exit, wait for all saves to complete
        saver.wait_all_complete()
        saver.stop()
    """

    def __init__(self, max_queue_size: int = 10):
        """
        Initialize the async episode saver.

        Args:
            max_queue_size: Maximum number of episodes that can be queued.
                           If full, new saves will fail immediately.
        """
        self.max_queue_size = max_queue_size
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # Thread-safe state tracking
        self._lock = threading.Lock()
        self._episode_index_counter = 0  # Atomic episode index allocation
        self._pending_saves: dict[int, EpisodeSaveTask] = {}
        self._completed_saves: dict[int, dict] = {}  # ep_index -> result
        self._failed_saves: dict[int, Exception] = {}

        # Statistics
        self._stats = {
            "total_queued": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_retries": 0,
        }

        logging.info("[AsyncEpisodeSaver] Initialized with max_queue_size=%d", max_queue_size)

    def allocate_next_index(self) -> int:
        """
        Atomically allocate the next episode index.

        This method should be called at the START of recording a new episode,
        BEFORE add_frame() is called. This ensures images are saved to the
        correct directory (episode_XXXXXX/) from the beginning.

        Returns:
            The allocated episode index
        """
        with self._lock:
            index = self._episode_index_counter
            self._episode_index_counter += 1
            logging.debug("[AsyncEpisodeSaver] Allocated episode index %d", index)
            return index

    def start(self, initial_episode_index: int = 0):
        """
        Start the background worker thread.

        Args:
            initial_episode_index: Starting episode index (usually meta.total_episodes)
        """
        with self._lock:
            self._episode_index_counter = initial_episode_index
            self.running = True

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncEpisodeSaver-Worker",
            daemon=True
        )
        self.worker_thread.start()
        logging.info("[AsyncEpisodeSaver] Background worker started (initial_ep_idx=%d)",
                    initial_episode_index)

    def queue_save(
        self,
        episode_buffer: dict,
        dataset: 'DoRobotDataset',
        record_cfg: 'RecordConfig',
        record_cmd: dict,
    ) -> EpisodeMetadata:
        """
        Queue an episode for asynchronous saving and return metadata immediately.

        This is the main API called by Record.save(). It:
        1. Uses the pre-allocated episode_index from the buffer (allocated at start of recording)
        2. Deep copies the episode buffer to avoid race conditions
        3. Queues the save task for background processing
        4. Returns metadata immediately (non-blocking)

        IMPORTANT: The episode_index must be pre-allocated using allocate_next_index()
        BEFORE recording starts. This ensures images are saved to the correct directory
        during add_frame() calls.

        Args:
            episode_buffer: Current episode data buffer (must have pre-allocated episode_index)
            dataset: DoRobotDataset instance
            record_cfg: Recording configuration
            record_cmd: Recording command metadata

        Returns:
            EpisodeMetadata with episode_index and queue position
        """
        start_time = time.time()

        # 1. Get the pre-allocated episode index from the buffer
        # This index was allocated at the START of recording via allocate_next_index()
        # so that images were saved to the correct directory during add_frame()
        episode_index = episode_buffer.get("episode_index")
        if episode_index is None:
            raise ValueError(
                "episode_buffer must have a pre-allocated episode_index. "
                "Call allocate_next_index() before creating the episode buffer."
            )

        with self._lock:
            queue_pos = self.save_queue.qsize()
            self._stats["total_queued"] += 1

        # 2. Create deep copy of episode_buffer (avoid shared state issues)
        # This is critical for thread safety!
        buffer_copy = copy.deepcopy(episode_buffer)

        # 4. Create save task
        task = EpisodeSaveTask(
            episode_index=episode_index,
            episode_buffer=buffer_copy,
            dataset=dataset,
            record_cfg=record_cfg,
            record_cmd=record_cmd,
            timestamp=time.time(),
        )

        # 5. Add to pending tracker
        with self._lock:
            self._pending_saves[episode_index] = task

        # 6. Queue for background processing
        task_queued = True
        try:
            self.save_queue.put(task, timeout=5.0)
            elapsed_ms = (time.time() - start_time) * 1000
            logging.info(
                "[AsyncEpisodeSaver] ✓ Queued episode %d (queue_pos=%d, frames=%d, elapsed=%.1fms)",
                episode_index, queue_pos, buffer_copy.get("size", 0), elapsed_ms
            )
        except queue.Full:
            logging.error("[AsyncEpisodeSaver] ✗ Save queue full! Episode %d dropped", episode_index)
            task_queued = False
            with self._lock:
                self._failed_saves[episode_index] = Exception("Queue full")
                if episode_index in self._pending_saves:
                    del self._pending_saves[episode_index]

        # 7. Return metadata immediately
        return EpisodeMetadata(
            episode_index=episode_index,
            last_record_episode_index=episode_index,
            estimated_frames=buffer_copy.get("size", 0),
            task_queued=task_queued,
            queue_position=queue_pos,
        )

    def _worker_loop(self):
        """
        Background thread main loop - processes save queue.

        Continuously pulls tasks from the queue and executes them.
        Runs until self.running is False and queue is empty.
        """
        logging.info("[AsyncEpisodeSaver] Worker loop started")

        while self.running or not self.save_queue.empty():
            try:
                # Get next task (with timeout to check running flag periodically)
                task = self.save_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._execute_save(task)
            except Exception as e:
                logging.error(
                    "[AsyncEpisodeSaver] Failed to save episode %d: %s\n%s",
                    task.episode_index, str(e), traceback.format_exc()
                )
                self._handle_save_failure(task, e)
            finally:
                self.save_queue.task_done()

        logging.info("[AsyncEpisodeSaver] Worker loop exited")

    def _execute_save(self, task: EpisodeSaveTask):
        """
        Execute the actual save operation (blocking).

        This method calls the existing dataset.save_episode() to ensure
        data format compatibility. The sequence is:
        1. Wait for image writer to finish
        2. Call dataset.save_episode() (parquet + video encoding)
        3. Update JSON metadata files
        4. Mark as completed

        Args:
            task: The save task to execute
        """
        ep_idx = task.episode_index
        logging.info("[AsyncEpisodeSaver] Starting save for episode %d", ep_idx)
        start_time = time.time()

        # NOTE: We do NOT wait for image_writer.wait_until_done() here!
        # That would block on ALL images (including images from subsequent episodes
        # that are still being recorded). Instead, save_episode() will wait for
        # only THIS episode's images via _wait_episode_images().

        # Step 1: Save episode to parquet + encode videos
        # This uses the EXISTING save_episode() method to ensure format compatibility
        logging.debug("[AsyncEpisodeSaver] Calling dataset.save_episode (ep %d)", ep_idx)
        save_start = time.time()
        actual_ep_idx = task.dataset.save_episode(episode_data=task.episode_buffer)
        save_time = time.time() - save_start
        logging.debug("[AsyncEpisodeSaver] save_episode done (ep %d, %.2fs)",
                     ep_idx, save_time)

        # Step 3: Update JSON metadata files (dataid.json, common_record.json)
        from operating_platform.utils.data_file import (
            update_dataid_json,
            update_common_record_json
        )

        update_dataid_json(task.record_cfg.root, actual_ep_idx, task.record_cmd)

        if actual_ep_idx == 0 and task.dataset.meta.total_episodes == 1:
            update_common_record_json(task.record_cfg.root, task.record_cmd)

        # Step 4: Mark as completed
        elapsed = time.time() - start_time
        result = {
            "episode_index": actual_ep_idx,
            "save_time_s": elapsed,
            "frames": task.episode_buffer.get("size", 0),
        }

        with self._lock:
            self._completed_saves[ep_idx] = result
            if ep_idx in self._pending_saves:
                del self._pending_saves[ep_idx]
            self._stats["total_completed"] += 1

        logging.info(
            "[AsyncEpisodeSaver] ✓ Episode %d saved successfully in %.2fs (frames=%d)",
            ep_idx, elapsed, task.episode_buffer.get("size", 0)
        )

    def _handle_save_failure(self, task: EpisodeSaveTask, error: Exception):
        """
        Handle failed save with retry logic.

        If retry count < max_retries, re-queue the task with exponential backoff.
        Otherwise, mark as permanently failed.

        Args:
            task: The failed save task
            error: The exception that caused the failure
        """
        ep_idx = task.episode_index

        if task.retry_count < task.max_retries:
            task.retry_count += 1
            backoff_time = 2 ** task.retry_count
            logging.warning(
                "[AsyncEpisodeSaver] Retry %d/%d for episode %d (backoff=%ds)",
                task.retry_count, task.max_retries, ep_idx, backoff_time
            )

            with self._lock:
                self._stats["total_retries"] += 1

            # Re-queue with exponential backoff
            time.sleep(backoff_time)
            self.save_queue.put(task)
        else:
            logging.error(
                "[AsyncEpisodeSaver] ✗ Episode %d failed after %d retries",
                ep_idx, task.max_retries
            )
            with self._lock:
                self._failed_saves[ep_idx] = error
                if ep_idx in self._pending_saves:
                    del self._pending_saves[ep_idx]
                self._stats["total_failed"] += 1

    def get_status(self) -> dict:
        """
        Get current status of all saves.

        Returns:
            Dictionary with queue size, pending/completed/failed counts, and stats
        """
        with self._lock:
            return {
                "running": self.running,
                "queue_size": self.save_queue.qsize(),
                "pending_count": len(self._pending_saves),
                "completed_count": len(self._completed_saves),
                "failed_count": len(self._failed_saves),
                "stats": self._stats.copy(),
                "pending_episodes": sorted(self._pending_saves.keys()),
                "failed_episodes": sorted(self._failed_saves.keys()),
            }

    def wait_all_complete(self, timeout: Optional[float] = None) -> bool:
        """
        Block until all queued saves are complete.

        Call this before exiting to ensure data integrity.

        Args:
            timeout: Maximum time to wait in seconds (None = infinite)

        Returns:
            True if all saves completed, False if timeout or error
        """
        logging.info("[AsyncEpisodeSaver] Waiting for all saves to complete...")
        start_time = time.time()

        try:
            if timeout:
                # NOTE: queue.join() doesn't have a timeout parameter!
                # We use polling instead to implement proper timeout behavior.
                poll_interval = 0.5
                while True:
                    # Check if all tasks are done
                    with self._lock:
                        pending = len(self._pending_saves)
                        queue_size = self.save_queue.qsize()

                    if pending == 0 and queue_size == 0:
                        break

                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logging.warning(
                            "[AsyncEpisodeSaver] Timeout waiting for saves after %.1fs "
                            "(pending=%d, queue=%d)",
                            elapsed, pending, queue_size
                        )
                        return False

                    time.sleep(poll_interval)
            else:
                self.save_queue.join()

            elapsed = time.time() - start_time
            logging.info("[AsyncEpisodeSaver] All saves completed in %.2fs", elapsed)
            return True
        except Exception as e:
            logging.error("[AsyncEpisodeSaver] Error waiting for completion: %s", e)
            return False

    def stop(self, wait_for_completion: bool = True):
        """
        Stop the background worker.

        Args:
            wait_for_completion: If True, wait for pending saves to finish
        """
        logging.info("[AsyncEpisodeSaver] Stopping (wait=%s)...", wait_for_completion)

        if wait_for_completion:
            self.wait_all_complete(timeout=300.0)  # 5 minute max wait

        self.running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10.0)
            if self.worker_thread.is_alive():
                logging.warning("[AsyncEpisodeSaver] Worker thread did not stop gracefully")

        # Log final statistics
        final_stats = self.get_status()
        logging.info(
            "[AsyncEpisodeSaver] Stopped. Stats: queued=%d completed=%d failed=%d retries=%d",
            final_stats["stats"]["total_queued"],
            final_stats["stats"]["total_completed"],
            final_stats["stats"]["total_failed"],
            final_stats["stats"]["total_retries"],
        )
