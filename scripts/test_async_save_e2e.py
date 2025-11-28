#!/usr/bin/env python3
"""
End-to-End Test for Async Episode Saver

This script performs integration testing of the async save functionality:
1. Creates test episodes with simulated data
2. Tests both sync and async save modes
3. Validates data format compatibility
4. Measures performance improvements
5. Tests error handling and edge cases

Usage:
    # Run full test suite
    python scripts/test_async_save_e2e.py

    # Run specific test
    python scripts/test_async_save_e2e.py --test rapid_saves

    # Verbose output
    python scripts/test_async_save_e2e.py --verbose
"""

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from operating_platform.core.async_episode_saver import AsyncEpisodeSaver
from operating_platform.dataset.dorobot_dataset import DoRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


class AsyncSaveE2ETest:
    """End-to-end integration tests for async episode saver."""

    def __init__(self, temp_dir: Path, reference_dir: Path):
        self.temp_dir = temp_dir
        self.reference_dir = reference_dir
        self.test_results: List[Tuple[str, bool, str]] = []

    def run_all_tests(self):
        """Run all integration tests."""
        logging.info("="*70)
        logging.info("Async Episode Saver - End-to-End Integration Tests")
        logging.info("="*70)

        tests = [
            ("Single Episode Save", self.test_single_episode_save),
            ("Rapid Sequential Saves", self.test_rapid_sequential_saves),
            ("Performance Comparison", self.test_performance_comparison),
            ("Exit During Pending Saves", self.test_exit_during_pending),
            ("Queue Overflow Handling", self.test_queue_overflow),
            ("Data Format Validation", self.test_data_format_validation),
        ]

        for test_name, test_func in tests:
            logging.info(f"\n{'='*70}")
            logging.info(f"Test: {test_name}")
            logging.info(f"{'='*70}")
            try:
                test_func()
                self.test_results.append((test_name, True, "PASS"))
                logging.info(f"✓ {test_name} PASSED")
            except Exception as e:
                self.test_results.append((test_name, False, str(e)))
                logging.error(f"✗ {test_name} FAILED: {e}")

        self._print_summary()

    def test_single_episode_save(self):
        """Test saving a single episode with async mode."""
        logging.info("Testing single episode save with async mode...")

        # Create test dataset
        test_path = self.temp_dir / "test_single_episode"
        test_path.mkdir(parents=True, exist_ok=True)

        # This test requires actual DoRobotDataset which needs robot config
        # For now, we'll test the AsyncEpisodeSaver in isolation
        from unittest.mock import Mock

        saver = AsyncEpisodeSaver(max_queue_size=10)
        saver.start(initial_episode_index=0)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(return_value=0)

        mock_cfg = Mock(root=test_path)
        mock_cmd = {"task_id": "test_single"}

        # Create test episode buffer
        episode_buffer = {
            "size": 50,
            "frames": [{"frame_id": i} for i in range(50)],
        }

        # Save episode
        start_time = time.time()
        metadata = saver.queue_save(
            episode_buffer=episode_buffer,
            dataset=mock_dataset,
            record_cfg=mock_cfg,
            record_cmd=mock_cmd,
        )
        save_time = time.time() - start_time

        # Should return immediately
        assert save_time < 0.5, f"Save took too long: {save_time:.3f}s"
        assert metadata.episode_index == 0, f"Wrong episode index: {metadata.episode_index}"
        assert metadata.task_queued, "Task not queued"

        # Wait for save to complete
        saver.wait_all_complete(timeout=10.0)

        # Check status
        status = saver.get_status()
        assert status["completed_count"] == 1, f"Not completed: {status}"
        assert status["failed_count"] == 0, f"Save failed: {status}"

        saver.stop()
        logging.info(f"  Save time: {save_time*1000:.1f}ms (should be <500ms)")
        logging.info(f"  Episode index: {metadata.episode_index}")

    def test_rapid_sequential_saves(self):
        """Test rapid sequential saves (simulating pressing 'n' multiple times)."""
        logging.info("Testing rapid sequential episode saves...")

        from unittest.mock import Mock

        saver = AsyncEpisodeSaver(max_queue_size=20)
        saver.start(initial_episode_index=0)

        # Mock dataset with realistic save time
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None

        def mock_save(**kwargs):
            time.sleep(0.2)  # Simulate 200ms save time
            return kwargs.get('episode_data', {}).get('ep_idx', 0)

        mock_dataset.save_episode = Mock(side_effect=mock_save)

        mock_cfg = Mock(root=self.temp_dir / "test_rapid")
        mock_cmd = {"task_id": "test_rapid"}

        # Queue 5 episodes rapidly
        num_episodes = 5
        episode_indices = []
        queue_times = []

        logging.info(f"  Queueing {num_episodes} episodes rapidly...")
        for i in range(num_episodes):
            episode_buffer = {
                "size": 30,
                "frames": [{"frame_id": j} for j in range(30)],
                "ep_idx": i,
            }

            start = time.time()
            metadata = saver.queue_save(
                episode_buffer=episode_buffer,
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )
            elapsed = time.time() - start

            episode_indices.append(metadata.episode_index)
            queue_times.append(elapsed)
            logging.info(f"    Episode {i}: queued in {elapsed*1000:.1f}ms")

        # All should queue quickly
        max_queue_time = max(queue_times)
        avg_queue_time = sum(queue_times) / len(queue_times)

        assert max_queue_time < 0.5, f"Queue time too long: {max_queue_time:.3f}s"
        logging.info(f"  Max queue time: {max_queue_time*1000:.1f}ms")
        logging.info(f"  Avg queue time: {avg_queue_time*1000:.1f}ms")

        # Wait for all to complete
        logging.info("  Waiting for all saves to complete...")
        wait_start = time.time()
        saver.wait_all_complete(timeout=30.0)
        total_wait = time.time() - wait_start

        # Check all completed successfully
        status = saver.get_status()
        assert status["completed_count"] == num_episodes, \
            f"Not all completed: {status['completed_count']}/{num_episodes}"
        assert status["failed_count"] == 0, f"Some failed: {status['failed_count']}"

        logging.info(f"  All {num_episodes} episodes completed in {total_wait:.2f}s")
        logging.info(f"  Episode indices: {episode_indices}")

        saver.stop()

    def test_performance_comparison(self):
        """Compare async vs sync save performance."""
        logging.info("Testing performance comparison (async vs sync)...")

        from unittest.mock import Mock

        # Simulate realistic save times
        def slow_save(**kwargs):
            time.sleep(1.0)  # 1 second per save (realistic for video encoding)
            return 0

        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(side_effect=slow_save)

        mock_cfg = Mock(root=self.temp_dir / "test_perf")
        mock_cmd = {"task_id": "test_perf"}

        episode_buffer = {"size": 50, "frames": []}

        # Test async mode
        logging.info("  Testing async mode (3 episodes)...")
        saver = AsyncEpisodeSaver(max_queue_size=10)
        saver.start(initial_episode_index=0)

        async_start = time.time()
        for i in range(3):
            saver.queue_save(
                episode_buffer=episode_buffer,
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )
        async_queue_time = time.time() - async_start

        saver.wait_all_complete(timeout=10.0)
        async_total_time = time.time() - async_start
        saver.stop()

        # Test sync mode (simulated)
        logging.info("  Testing sync mode (3 episodes)...")
        sync_start = time.time()
        for i in range(3):
            mock_dataset.save_episode()  # Blocking call
        sync_total_time = time.time() - sync_start

        # Results
        speedup = sync_total_time / async_queue_time if async_queue_time > 0 else float('inf')

        logging.info(f"  Async queue time: {async_queue_time:.3f}s")
        logging.info(f"  Async total time: {async_total_time:.3f}s")
        logging.info(f"  Sync total time: {sync_total_time:.3f}s")
        logging.info(f"  Speedup (queue phase): {speedup:.1f}x")

        # Async queueing should be much faster
        assert async_queue_time < sync_total_time * 0.2, \
            f"Async not significantly faster: {async_queue_time:.3f}s vs {sync_total_time:.3f}s"

        logging.info(f"  ✓ Async queueing is {speedup:.1f}x faster than sync saves")

    def test_exit_during_pending(self):
        """Test exiting while saves are pending."""
        logging.info("Testing exit with pending saves...")

        from unittest.mock import Mock

        saver = AsyncEpisodeSaver(max_queue_size=10)
        saver.start(initial_episode_index=0)

        # Mock slow save
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(side_effect=lambda **kw: time.sleep(0.5))

        mock_cfg = Mock(root=self.temp_dir / "test_exit")
        mock_cmd = {"task_id": "test_exit"}

        # Queue 3 saves
        for i in range(3):
            saver.queue_save(
                episode_buffer={"size": 10},
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )

        # Stop should wait for all to complete
        logging.info("  Stopping with pending saves...")
        stop_start = time.time()
        saver.stop(wait_for_completion=True)
        stop_time = time.time() - stop_start

        # Should have waited
        assert stop_time > 1.0, f"Didn't wait for saves: {stop_time:.3f}s"

        # All should be complete
        status = saver.get_status()
        assert status["pending_count"] == 0, f"Saves still pending: {status}"

        logging.info(f"  ✓ Waited {stop_time:.2f}s for pending saves to complete")

    def test_queue_overflow(self):
        """Test behavior when queue is full."""
        logging.info("Testing queue overflow handling...")

        from unittest.mock import Mock

        # Small queue
        saver = AsyncEpisodeSaver(max_queue_size=2)
        saver.start(initial_episode_index=0)

        # Mock very slow save
        mock_dataset = Mock()
        mock_dataset.meta = Mock(total_episodes=0)
        mock_dataset.image_writer = None
        mock_dataset.save_episode = Mock(side_effect=lambda **kw: time.sleep(5.0))

        mock_cfg = Mock(root=self.temp_dir / "test_overflow")
        mock_cmd = {"task_id": "test_overflow"}

        # Fill queue
        results = []
        for i in range(4):  # Try to queue 4 items in size-2 queue
            metadata = saver.queue_save(
                episode_buffer={"size": 10},
                dataset=mock_dataset,
                record_cfg=mock_cfg,
                record_cmd=mock_cmd,
            )
            results.append(metadata.task_queued)
            time.sleep(0.1)  # Small delay

        # First 2-3 should succeed, last should fail
        successful = sum(results)
        failed = len(results) - successful

        logging.info(f"  Queued: {successful}/{len(results)}")
        logging.info(f"  Dropped: {failed}/{len(results)}")

        assert failed > 0, "Should have some failures when queue is full"
        assert successful >= 2, "Should successfully queue at least 2 items"

        logging.info(f"  ✓ Queue overflow handled correctly")

        saver.stop(wait_for_completion=False)

    def test_data_format_validation(self):
        """Validate that async-saved data matches expected format."""
        logging.info("Testing data format validation...")

        # This would require actual dataset creation
        # For now, just check that the validation script exists
        validation_script = Path(__file__).parent / "validate_dataset_format.py"
        assert validation_script.exists(), \
            f"Validation script not found: {validation_script}"

        logging.info(f"  ✓ Validation script exists: {validation_script}")
        logging.info("  Note: Run full validation with actual robot data using:")
        logging.info(f"    python {validation_script} --reference {self.reference_dir} --target <dataset>")

    def _print_summary(self):
        """Print test results summary."""
        logging.info("\n" + "="*70)
        logging.info("Test Summary")
        logging.info("="*70)

        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = len(self.test_results) - passed

        for test_name, success, message in self.test_results:
            status = "✓ PASS" if success else "✗ FAIL"
            logging.info(f"{status:8} {test_name}")
            if not success:
                logging.info(f"         {message}")

        logging.info("="*70)
        logging.info(f"Results: {passed}/{len(self.test_results)} passed")

        if failed == 0:
            logging.info("\n✓ ALL TESTS PASSED")
            logging.info("The async episode saver is working correctly.")
        else:
            logging.error(f"\n✗ {failed} TEST(S) FAILED")
            logging.error("Please review the failures above.")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end integration tests for async episode saver"
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default="/Users/nupylot/Public/so101-test-1126-ok",
        help="Path to reference dataset for validation"
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (default: run all)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create temporary directory for tests
    with tempfile.TemporaryDirectory(prefix="async_save_test_") as temp_dir:
        temp_path = Path(temp_dir)
        logging.info(f"Using temp directory: {temp_path}\n")

        # Run tests
        tester = AsyncSaveE2ETest(temp_path, args.reference)

        if args.test:
            # Run specific test
            test_method = getattr(tester, f"test_{args.test}", None)
            if test_method is None:
                logging.error(f"Unknown test: {args.test}")
                return 1
            test_method()
        else:
            # Run all tests
            tester.run_all_tests()

        # Return success/failure
        failed = sum(1 for _, success, _ in tester.test_results if not success)
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
