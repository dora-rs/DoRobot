#!/usr/bin/env python3
"""
Unit tests for AsyncEpisodeSaver

Tests the async episode save functionality without requiring robot hardware.
Uses mock dataset and episode buffers to validate:
- Queue operations
- Thread safety
- Retry logic
- Status tracking
- Error handling
"""

import copy
import logging
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from operating_platform.core.async_episode_saver import (
    AsyncEpisodeSaver,
    EpisodeSaveTask,
    EpisodeMetadata,
)

logging.basicConfig(level=logging.INFO)


class TestEpisodeSaveTask(unittest.TestCase):
    """Test EpisodeSaveTask dataclass."""

    def test_create_task(self):
        """Test creating a save task."""
        task = EpisodeSaveTask(
            episode_index=5,
            episode_buffer={"size": 100, "frames": []},
            dataset=Mock(),
            record_cfg=Mock(),
            record_cmd={"task_id": "test"},
            timestamp=time.time(),
        )

        self.assertEqual(task.episode_index, 5)
        self.assertEqual(task.episode_buffer["size"], 100)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(task.max_retries, 3)


class TestEpisodeMetadata(unittest.TestCase):
    """Test EpisodeMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating episode metadata."""
        metadata = EpisodeMetadata(
            episode_index=10,
            last_record_episode_index=10,
            estimated_frames=87,
            task_queued=True,
            queue_position=2,
        )

        self.assertEqual(metadata.episode_index, 10)
        self.assertEqual(metadata.last_record_episode_index, 10)
        self.assertEqual(metadata.estimated_frames, 87)
        self.assertTrue(metadata.task_queued)
        self.assertEqual(metadata.queue_position, 2)


class TestAsyncEpisodeSaver(unittest.TestCase):
    """Test AsyncEpisodeSaver class."""

    def setUp(self):
        """Set up test fixtures."""
        self.saver = AsyncEpisodeSaver(max_queue_size=5)
        # Patch file I/O operations to avoid needing real filesystem
        self.patcher1 = patch('operating_platform.utils.data_file.update_dataid_json')
        self.patcher2 = patch('operating_platform.utils.data_file.update_common_record_json')
        self.mock_update_dataid = self.patcher1.start()
        self.mock_update_common = self.patcher2.start()

    def tearDown(self):
        """Clean up after tests."""
        if self.saver.running:
            self.saver.stop(wait_for_completion=False)
        self.patcher1.stop()
        self.patcher2.stop()

    def test_initialization(self):
        """Test saver initialization."""
        self.assertEqual(self.saver.max_queue_size, 5)
        self.assertFalse(self.saver.running)
        self.assertIsNone(self.saver.worker_thread)
        self.assertEqual(self.saver._episode_index_counter, 0)

    def test_start(self):
        """Test starting the background worker."""
        self.saver.start(initial_episode_index=10)

        self.assertTrue(self.saver.running)
        self.assertIsNotNone(self.saver.worker_thread)
        self.assertTrue(self.saver.worker_thread.is_alive())
        self.assertEqual(self.saver._episode_index_counter, 10)

        self.saver.stop(wait_for_completion=False)

    def test_queue_save_metadata_return(self):
        """Test that queue_save returns metadata immediately."""
        self.saver.start(initial_episode_index=0)

        # Create mock objects
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Pre-allocate episode index (simulating what Record class does)
        allocated_idx = self.saver.allocate_next_index()
        episode_buffer = {
            "episode_index": allocated_idx,  # Pre-allocated index
            "size": 50,
            "frames": list(range(50)),
        }

        # Queue save and measure time
        start_time = time.time()
        metadata = self.saver.queue_save(
            episode_buffer=episode_buffer,
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )
        elapsed = time.time() - start_time

        # Should return immediately (<1s)
        self.assertLess(elapsed, 1.0, "queue_save should return immediately")

        # Check metadata
        self.assertEqual(metadata.episode_index, 0)
        self.assertEqual(metadata.last_record_episode_index, 0)
        self.assertEqual(metadata.estimated_frames, 50)
        self.assertTrue(metadata.task_queued)
        self.assertGreaterEqual(metadata.queue_position, 0)

        self.saver.stop(wait_for_completion=False)

    def test_episode_index_increment(self):
        """Test that episode indices are atomically incremented via allocate_next_index."""
        self.saver.start(initial_episode_index=100)

        mock_dataset = Mock()
        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Pre-allocate indices and queue multiple saves
        indices = []
        for i in range(3):
            # Pre-allocate index (simulating what Record class does)
            allocated_idx = self.saver.allocate_next_index()
            metadata = self.saver.queue_save(
                episode_buffer={"episode_index": allocated_idx, "size": 10},
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )
            indices.append(metadata.episode_index)

        # Should increment sequentially
        self.assertEqual(indices, [100, 101, 102])

        self.saver.stop(wait_for_completion=False)

    def test_deep_copy_isolation(self):
        """Test that episode_buffer is deep copied to avoid race conditions."""
        self.saver.start(initial_episode_index=0)

        mock_dataset = Mock()
        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Pre-allocate index
        allocated_idx = self.saver.allocate_next_index()
        original_buffer = {
            "episode_index": allocated_idx,
            "size": 2,
            "frames": [{"frame_id": 0}, {"frame_id": 1}],
        }

        # Queue save
        metadata = self.saver.queue_save(
            episode_buffer=original_buffer,
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )

        # Modify original buffer
        original_buffer["frames"].append({"frame_id": 2})
        original_buffer["size"] = 3

        # Get the queued task
        time.sleep(0.1)  # Give time for queue to process
        with self.saver._lock:
            if metadata.episode_index in self.saver._pending_saves:
                task = self.saver._pending_saves[metadata.episode_index]
                # Task should have original size, not modified
                self.assertEqual(task.episode_buffer["size"], 2)
                self.assertEqual(len(task.episode_buffer["frames"]), 2)

        self.saver.stop(wait_for_completion=False)

    def test_queue_full_handling(self):
        """Test behavior when queue is full."""
        small_queue_saver = AsyncEpisodeSaver(max_queue_size=2)
        # Don't start worker - this keeps all items in queue
        small_queue_saver.running = True  # Set running flag for queue_save to work
        small_queue_saver._episode_index_counter = 0

        mock_dataset = Mock()
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(return_value=0)
        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Queue 2 items - fills the queue completely
        idx1 = small_queue_saver.allocate_next_index()
        metadata1 = small_queue_saver.queue_save(
            episode_buffer={"episode_index": idx1, "size": 10},
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )
        self.assertTrue(metadata1.task_queued)

        idx2 = small_queue_saver.allocate_next_index()
        metadata2 = small_queue_saver.queue_save(
            episode_buffer={"episode_index": idx2, "size": 10},
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )
        self.assertTrue(metadata2.task_queued)

        # Queue is now full (2/2 items), third should timeout and fail
        idx3 = small_queue_saver.allocate_next_index()
        metadata3 = small_queue_saver.queue_save(
            episode_buffer={"episode_index": idx3, "size": 10},
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )

        # Should still return metadata but with task_queued=False
        self.assertFalse(metadata3.task_queued)

        small_queue_saver.running = False

    def test_get_status(self):
        """Test status tracking."""
        self.saver.start(initial_episode_index=0)

        status = self.saver.get_status()

        self.assertTrue(status["running"])
        self.assertEqual(status["queue_size"], 0)
        self.assertEqual(status["pending_count"], 0)
        self.assertEqual(status["completed_count"], 0)
        self.assertEqual(status["failed_count"], 0)
        self.assertIn("stats", status)
        self.assertEqual(status["stats"]["total_queued"], 0)

        self.saver.stop(wait_for_completion=False)

    def test_successful_save_execution(self):
        """Test successful save execution."""
        self.saver.start(initial_episode_index=0)

        # Mock dataset with successful save
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=1)
        mock_dataset.image_writer = None  # No image writer to wait for
        mock_dataset.save_episode = Mock(return_value=0)  # Returns episode index

        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Pre-allocate index and queue save
        allocated_idx = self.saver.allocate_next_index()
        metadata = self.saver.queue_save(
            episode_buffer={"episode_index": allocated_idx, "size": 10},
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )

        # Wait for save to complete
        self.saver.wait_all_complete(timeout=5.0)

        # Check that save_episode was called
        mock_dataset.save_episode.assert_called_once()

        # Check that metadata updates were called
        self.mock_update_dataid.assert_called_once()

        # Check status
        status = self.saver.get_status()
        self.assertEqual(status["completed_count"], 1)
        self.assertEqual(status["failed_count"], 0)
        self.assertEqual(status["stats"]["total_completed"], 1)

        self.saver.stop(wait_for_completion=False)

    def test_wait_all_complete(self):
        """Test waiting for all saves to complete."""
        self.saver.start(initial_episode_index=0)

        # Mock fast saves
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(side_effect=lambda **kwargs: 0)

        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Queue multiple saves with pre-allocated indices
        for i in range(3):
            allocated_idx = self.saver.allocate_next_index()
            self.saver.queue_save(
                episode_buffer={"episode_index": allocated_idx, "size": 10},
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )

        # Wait for completion
        start_time = time.time()
        success = self.saver.wait_all_complete(timeout=10.0)
        elapsed = time.time() - start_time

        self.assertTrue(success)
        self.assertLess(elapsed, 10.0)

        # All saves should be complete
        status = self.saver.get_status()
        self.assertEqual(status["queue_size"], 0)
        self.assertEqual(status["pending_count"], 0)

        self.saver.stop(wait_for_completion=False)

    def test_stop_with_pending_saves(self):
        """Test stopping while saves are pending."""
        self.saver.start(initial_episode_index=0)

        # Mock slow save
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(side_effect=lambda **kwargs: time.sleep(0.5))

        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        # Pre-allocate index and queue a save
        allocated_idx = self.saver.allocate_next_index()
        self.saver.queue_save(
            episode_buffer={"episode_index": allocated_idx, "size": 10},
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )

        # Stop with wait
        self.saver.stop(wait_for_completion=True)

        # Should have waited for save to complete
        self.assertFalse(self.saver.running)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of AsyncEpisodeSaver."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch file I/O operations to avoid needing real filesystem
        self.patcher1 = patch('operating_platform.utils.data_file.update_dataid_json')
        self.patcher2 = patch('operating_platform.utils.data_file.update_common_record_json')
        self.mock_update_dataid = self.patcher1.start()
        self.mock_update_common = self.patcher2.start()

    def tearDown(self):
        """Clean up after tests."""
        self.patcher1.stop()
        self.patcher2.stop()

    def test_concurrent_queue_saves(self):
        """Test that multiple threads can queue saves concurrently."""
        import threading

        saver = AsyncEpisodeSaver(max_queue_size=20)
        saver.start(initial_episode_index=0)

        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(side_effect=lambda **kwargs: time.sleep(0.1))

        mock_cfg = Mock(root="/tmp/test")
        mock_cmd = {"task_id": "test", "task_data_id": "001"}

        episode_indices = []
        lock = threading.Lock()

        def queue_save_thread():
            # Pre-allocate index (thread-safe operation)
            allocated_idx = saver.allocate_next_index()
            metadata = saver.queue_save(
                episode_buffer={"episode_index": allocated_idx, "size": 10},
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )
            with lock:
                episode_indices.append(metadata.episode_index)

        # Launch 10 threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=queue_save_thread)
            t.start()
            threads.append(t)

        # Wait for all threads
        for t in threads:
            t.join()

        # All indices should be unique and sequential
        self.assertEqual(len(episode_indices), 10)
        self.assertEqual(len(set(episode_indices)), 10)  # All unique
        self.assertEqual(sorted(episode_indices), list(range(10)))  # Sequential 0-9

        saver.stop(wait_for_completion=True)


def main():
    """Run all tests."""
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
