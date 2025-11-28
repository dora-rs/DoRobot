# Detailed Design Plan: Asynchronous Episode Save System

## Problem Analysis

### Current Bottleneck

**Current Blocking Behavior:**
1. When user presses 'n' (new episode) or 'e' (exit), `record.save()` is called synchronously
2. `save_episode()` → `_wait_image_writer()` → `image_writer.wait_until_done()` blocks until ALL images are written
3. Video encoding via `encode_episode_videos()` also blocks
4. User must wait 10-30+ seconds before starting next data collection cycle

**Key Requirement:**
- Return essential metadata immediately (episode_index, last_record_episode_index)
- Execute actual save operations in background
- Allow next data collection cycle to start without waiting

---

## Architecture Design

### 1. Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Recording Loop                       │
│  (operating_platform/core/main.py::record_loop)             │
└────────────┬────────────────────────────────────────────────┘
             │
             │ Press 'n' or 'e'
             ▼
┌────────────────────────────────────────────────────────────┐
│         AsyncEpisodeSaver (NEW)                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Reserve episode_index (atomic)                   │  │
│  │  2. Return metadata immediately                       │  │
│  │  3. Queue save task to background thread             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Background Worker Thread:                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  - Process save queue (FIFO)                         │  │
│  │  - Execute: wait_image_writer() → save_episode() →  │  │
│  │             encode_videos() → update_metadata()      │  │
│  │  - Handle errors with retry logic                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
             │
             │ Completed episodes
             ▼
┌────────────────────────────────────────────────────────────┐
│          Status Tracker & Error Handler                    │
│  - Track pending/completed/failed saves                    │
│  - Log progress and errors                                 │
│  - Provide status query API                                │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Implementation Plan

### Phase 1: Create AsyncEpisodeSaver Class

**File:** `operating_platform/core/async_episode_saver.py` (NEW)

```python
import queue
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
import time


@dataclass
class EpisodeSaveTask:
    """Represents a single episode save operation"""
    episode_index: int
    episode_buffer: dict  # Deep copy of current episode buffer
    dataset: 'DoRobotDataset'
    record_cfg: 'RecordConfig'
    record_cmd: dict
    timestamp: float  # When task was created
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class EpisodeMetadata:
    """Immediate return value when save is requested"""
    episode_index: int
    last_record_episode_index: int
    estimated_frames: int
    task_queued: bool
    queue_position: int  # Position in save queue


class AsyncEpisodeSaver:
    """
    Handles asynchronous episode saving with background worker thread.
    Provides immediate metadata return while actual save happens in background.
    """

    def __init__(self, max_queue_size: int = 10):
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

    def start(self, initial_episode_index: int = 0):
        """Start the background worker thread"""
        with self._lock:
            self._episode_index_counter = initial_episode_index
            self.running = True

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncEpisodeSaver-Worker",
            daemon=True
        )
        self.worker_thread.start()
        logging.info("[AsyncEpisodeSaver] Background worker started")

    def queue_save(
        self,
        episode_buffer: dict,
        dataset: 'DoRobotDataset',
        record_cfg: 'RecordConfig',
        record_cmd: dict,
    ) -> EpisodeMetadata:
        """
        Queue an episode for saving and return metadata immediately.

        This is the main API called by record.save()
        """
        # 1. Atomically reserve next episode index
        with self._lock:
            episode_index = self._episode_index_counter
            self._episode_index_counter += 1
            queue_pos = self.save_queue.qsize()
            self._stats["total_queued"] += 1

        # 2. Create deep copy of episode_buffer (avoid shared state issues)
        import copy
        buffer_copy = copy.deepcopy(episode_buffer)

        # 3. Create save task
        task = EpisodeSaveTask(
            episode_index=episode_index,
            episode_buffer=buffer_copy,
            dataset=dataset,
            record_cfg=record_cfg,
            record_cmd=record_cmd,
            timestamp=time.time(),
        )

        # 4. Add to pending tracker
        with self._lock:
            self._pending_saves[episode_index] = task

        # 5. Queue for background processing
        try:
            self.save_queue.put(task, timeout=5.0)
            logging.info(f"[AsyncEpisodeSaver] Queued episode {episode_index} for saving (queue pos: {queue_pos})")
        except queue.Full:
            logging.error(f"[AsyncEpisodeSaver] Save queue full! Episode {episode_index} dropped")
            with self._lock:
                self._failed_saves[episode_index] = Exception("Queue full")
                del self._pending_saves[episode_index]

        # 6. Return metadata immediately
        return EpisodeMetadata(
            episode_index=episode_index,
            last_record_episode_index=episode_index,
            estimated_frames=buffer_copy.get("size", 0),
            task_queued=True,
            queue_position=queue_pos,
        )

    def _worker_loop(self):
        """Background thread main loop - processes save queue"""
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
                logging.error(f"[AsyncEpisodeSaver] Failed to save episode {task.episode_index}: {e}")
                self._handle_save_failure(task, e)
            finally:
                self.save_queue.task_done()

        logging.info("[AsyncEpisodeSaver] Worker loop exited")

    def _execute_save(self, task: EpisodeSaveTask):
        """Execute the actual save operation (blocking)"""
        ep_idx = task.episode_index
        logging.info(f"[AsyncEpisodeSaver] Starting save for episode {ep_idx}")
        start_time = time.time()

        # Step 1: Wait for image writer to finish writing all images
        if task.dataset.image_writer is not None:
            logging.debug(f"[AsyncEpisodeSaver] Waiting for image writer (ep {ep_idx})")
            task.dataset.image_writer.wait_until_done()

        # Step 2: Save episode to parquet + encode videos
        logging.debug(f"[AsyncEpisodeSaver] Calling dataset.save_episode (ep {ep_idx})")
        actual_ep_idx = task.dataset.save_episode(episode_data=task.episode_buffer)

        # Step 3: Update JSON metadata files
        from operating_platform.utils.data_file import update_dataid_json, update_common_record_json
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

        logging.info(f"[AsyncEpisodeSaver] ✓ Episode {ep_idx} saved successfully in {elapsed:.2f}s")

    def _handle_save_failure(self, task: EpisodeSaveTask, error: Exception):
        """Handle failed save with retry logic"""
        ep_idx = task.episode_index

        if task.retry_count < task.max_retries:
            task.retry_count += 1
            logging.warning(f"[AsyncEpisodeSaver] Retry {task.retry_count}/{task.max_retries} for episode {ep_idx}")

            with self._lock:
                self._stats["total_retries"] += 1

            # Re-queue with exponential backoff
            time.sleep(2 ** task.retry_count)
            self.save_queue.put(task)
        else:
            logging.error(f"[AsyncEpisodeSaver] ✗ Episode {ep_idx} failed after {task.max_retries} retries")
            with self._lock:
                self._failed_saves[ep_idx] = error
                if ep_idx in self._pending_saves:
                    del self._pending_saves[ep_idx]
                self._stats["total_failed"] += 1

    def get_status(self) -> dict:
        """Get current status of all saves"""
        with self._lock:
            return {
                "running": self.running,
                "queue_size": self.save_queue.qsize(),
                "pending_count": len(self._pending_saves),
                "completed_count": len(self._completed_saves),
                "failed_count": len(self._failed_saves),
                "stats": self._stats.copy(),
                "pending_episodes": list(self._pending_saves.keys()),
                "failed_episodes": list(self._failed_saves.keys()),
            }

    def wait_all_complete(self, timeout: Optional[float] = None) -> bool:
        """
        Block until all queued saves are complete.
        Call this before exiting to ensure data integrity.
        """
        logging.info("[AsyncEpisodeSaver] Waiting for all saves to complete...")
        try:
            self.save_queue.join()
            logging.info("[AsyncEpisodeSaver] All saves completed")
            return True
        except Exception as e:
            logging.error(f"[AsyncEpisodeSaver] Error waiting for completion: {e}")
            return False

    def stop(self, wait_for_completion: bool = True):
        """Stop the background worker"""
        logging.info("[AsyncEpisodeSaver] Stopping...")

        if wait_for_completion:
            self.wait_all_complete()

        self.running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10.0)

        logging.info(f"[AsyncEpisodeSaver] Stopped. Final stats: {self._stats}")
```

---

### Phase 2: Modify Record Class

**File:** `operating_platform/core/record.py`

**Changes:**

1. Add `async_saver` instance variable
2. Modify `save()` method to return immediately
3. Add `save_async()` and `save_sync()` methods

```python
# In Record.__init__():
from operating_platform.core.async_episode_saver import AsyncEpisodeSaver

def __init__(self, ...):
    # ... existing code ...

    # NEW: Create async saver
    self.async_saver = AsyncEpisodeSaver(max_queue_size=10)
    self.use_async_save = True  # Flag to enable/disable async saving


# NEW: Async version of save
def save_async(self) -> EpisodeMetadata:
    """
    Queue episode for asynchronous saving.
    Returns metadata immediately without blocking.
    """
    print(f"[Record] Queueing episode for async save...")

    # Queue save task
    metadata = self.async_saver.queue_save(
        episode_buffer=self.dataset.episode_buffer,
        dataset=self.dataset,
        record_cfg=self.record_cfg,
        record_cmd=self.record_cmd,
    )

    # Update local state immediately
    self.last_record_episode_index = metadata.episode_index
    self.record_complete = True

    print(f"[Record] Episode {metadata.episode_index} queued (pos: {metadata.queue_position})")
    return metadata


# MODIFIED: Keep original synchronous version
def save_sync(self) -> dict:
    """Original synchronous save (renamed from save)"""
    # ... existing save() implementation ...


# NEW: Unified save method with mode selection
def save(self) -> EpisodeMetadata | dict:
    """
    Save episode - async by default, fallback to sync if needed.
    """
    if self.use_async_save:
        return self.save_async()
    else:
        return self.save_sync()


# NEW: Start async saver when recording starts
def start(self):
    self.thread.start()
    self.running = True

    # Start async saver
    if self.use_async_save:
        initial_ep_idx = self.dataset.meta.total_episodes
        self.async_saver.start(initial_episode_index=initial_ep_idx)


# MODIFIED: Stop async saver when recording stops
def stop(self):
    if self.running == True:
        self.running = False
        self.thread.join()
        self.dataset.stop_audio_writer()

    # Stop async saver (wait for pending saves)
    if self.use_async_save:
        status = self.async_saver.get_status()
        if status["pending_count"] > 0:
            print(f"[Record] Waiting for {status['pending_count']} pending saves...")
            self.async_saver.wait_all_complete(timeout=60.0)
```

---

### Phase 3: Modify main.py Recording Loop

**File:** `operating_platform/core/main.py`

**Changes:**

```python
def record_loop(cfg: ControlPipelineConfig, daemon: Daemon):
    # ... setup code ...

    while True:
        # ... recording session ...

        while True:
            daemon.update()
            observation = daemon.get_observation()

            # ... display images ...

            key = cv2.waitKey(10)
            if key in [ord('n'), ord('N')]:
                logging.info("Ending current episode...")
                break
            elif key in [ord('e'), ord('E')]:
                logging.info("Stopping recording and exiting...")
                record.stop()

                # MODIFIED: Async save returns immediately
                metadata = record.save()
                logging.info(f"Episode {metadata.episode_index} queued for saving")

                # NEW: Wait for all pending saves before exiting
                if hasattr(record, 'async_saver'):
                    status = record.async_saver.get_status()
                    logging.info(f"Waiting for {status['pending_count']} pending saves...")
                    record.async_saver.wait_all_complete(timeout=120.0)

                    # Print final status
                    final_status = record.async_saver.get_status()
                    logging.info(f"Save stats: {final_status['stats']}")

                return

        # MODIFIED: Save current episode (now non-blocking)
        record.stop()
        metadata = record.save()

        # NEW: Log save status instead of blocking
        logging.info(f"Episode {metadata.episode_index} queued "
                    f"(queue pos: {metadata.queue_position})")

        # NEW: Optionally check for save errors from previous episodes
        if hasattr(record, 'async_saver'):
            status = record.async_saver.get_status()
            if status["failed_count"] > 0:
                logging.warning(f"⚠ {status['failed_count']} episodes failed to save: "
                               f"{status['failed_episodes']}")

        # Continue to reset/next episode immediately
        logging.info("*" * 30)
        logging.info("Resetting environment - Press 'p' to proceed")
        # ... reset logic ...
```

---

## 3. Thread Safety Considerations

### Critical Shared Resources

1. `dataset.meta` (total_episodes, total_frames counters)
2. `dataset.episode_buffer`
3. `dataset.image_writer` queue

### Safety Measures

```python
# In DoRobotDataset
class DoRobotDataset:
    def __init__(self, ...):
        # NEW: Add lock for thread-safe metadata updates
        self._meta_lock = threading.Lock()

    def save_episode(self, episode_data: dict | None = None) -> int:
        # Acquire lock when updating metadata
        with self._meta_lock:
            # ... metadata updates ...
            self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
```

### Episode Buffer Isolation

- Deep copy `episode_buffer` when queuing save task
- Reset buffer immediately after copying
- Each save task operates on independent copy

---

## 4. Error Handling Strategy

### Scenarios & Solutions

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Image writer queue full | Check queue size before save | Increase threads or retry |
| Disk space full | Catch IOError during save | Alert user, pause recording |
| Save queue full | Queue.Full exception | Drop episode with error log |
| Save timeout (>5 min) | Track task timestamp | Kill task, retry once |
| Crash during save | Check pending on restart | Replay from buffer backup |

### Implementation

```python
# In AsyncEpisodeSaver._execute_save()
try:
    # Set timeout for entire save operation
    import signal
    signal.alarm(300)  # 5 minute timeout

    # ... save operations ...

    signal.alarm(0)  # Cancel alarm
except IOError as e:
    if "No space left" in str(e):
        logging.critical("DISK FULL! Stopping recording")
        # Trigger emergency stop
    raise
except Exception as e:
    logging.error(f"Save failed: {e}", exc_info=True)
    raise
```

---

## 5. Testing Plan

### Unit Tests

```python
# tests/test_async_episode_saver.py
def test_immediate_return():
    """Verify save() returns in <100ms"""
    saver = AsyncEpisodeSaver()
    start = time.time()
    metadata = saver.queue_save(...)
    elapsed = time.time() - start
    assert elapsed < 0.1  # Must return within 100ms

def test_episode_index_allocation():
    """Verify atomic episode index incrementing"""
    # Concurrent save requests should get unique indices

def test_queue_overflow():
    """Verify behavior when queue is full"""

def test_save_ordering():
    """Verify episodes saved in FIFO order"""
```

### Integration Tests

```python
def test_rapid_episodes():
    """Simulate rapid 'n' presses (5 episodes in 10 seconds)"""

def test_exit_during_save():
    """Verify graceful shutdown during pending saves"""

def test_concurrent_image_writing():
    """Verify image writer compatibility with async saves"""
```

### Stress Tests

```python
def test_100_episodes_async():
    """Queue 100 episodes and verify all complete"""

def test_disk_full_recovery():
    """Simulate disk full error and verify error handling"""
```

---

## 6. Configuration & Monitoring

### New Config Options

```python
@dataclass
class RecordConfig:
    # ... existing fields ...

    # NEW: Async save options
    use_async_save: bool = True
    async_save_queue_size: int = 10
    async_save_timeout_s: int = 300
    async_save_max_retries: int = 3
```

### Monitoring Dashboard

```python
# Print status periodically
def print_save_status(async_saver: AsyncEpisodeSaver):
    status = async_saver.get_status()
    print(f"""
    ╔══════════════════════════════════════╗
    ║     Async Save Status                ║
    ╠══════════════════════════════════════╣
    ║ Queue Size:     {status['queue_size']:3d}              ║
    ║ Pending:        {status['pending_count']:3d}              ║
    ║ Completed:      {status['completed_count']:3d}              ║
    ║ Failed:         {status['failed_count']:3d}              ║
    ║ Total Retries:  {status['stats']['total_retries']:3d}              ║
    ╚══════════════════════════════════════╝
    """)
```

---

## 7. Migration Path

### Phase 1: Add Async System (Backward Compatible)
- Add AsyncEpisodeSaver class
- Add `save_async()` to Record
- Keep `save_sync()` as default
- Users opt-in via config flag

### Phase 2: Beta Testing
- Enable async save for test users
- Monitor for issues
- Collect performance metrics

### Phase 3: Make Async Default
- Switch default to `use_async_save=True`
- Keep sync option for debugging

### Phase 4: Deprecate Sync (Optional)
- Remove sync save after 3 months
- Simplify codebase

---

## 8. Expected Performance Improvements

### Current Behavior
- Press 'n' → Wait 15-30s → Start next episode
- Press 'e' → Wait 20-40s → Exit

### After Async Save
- Press 'n' → Wait <0.1s → Start next episode immediately
- Press 'e' → Wait ~5s (only for final cleanup) → Exit

### Throughput Improvement
- Current: ~2-3 episodes/minute (limited by save time)
- Async: ~6-10 episodes/minute (limited only by recording speed)

### User Experience
- Eliminates frustrating wait times
- Allows rapid iteration during data collection
- Background saves complete transparently

---

## 9. Rollback Strategy

If issues arise:
1. Set `use_async_save=False` in config
2. Falls back to synchronous saves
3. No data loss (async system is additive)

---

## 10. Future Enhancements

1. **Priority Queue**: Save most recent episodes first
2. **Compression**: Compress videos in background
3. **Upload to Cloud**: Auto-upload completed episodes to S3/HuggingFace
4. **Progress Bar**: Show real-time save progress
5. **Checkpoint Recovery**: Resume failed saves after crash

---

## Implementation Checklist

- [ ] Create `async_episode_saver.py` with AsyncEpisodeSaver class
- [ ] Add threading locks to DoRobotDatasetMetadata
- [ ] Modify Record class with save_async() method
- [ ] Update main.py recording loop
- [ ] Add configuration options to RecordConfig
- [ ] Write unit tests for AsyncEpisodeSaver
- [ ] Write integration tests for full record→save→reset cycle
- [ ] Add error handling and retry logic
- [ ] Implement monitoring/status dashboard
- [ ] Test with real robot data collection
- [ ] Document new async save behavior
- [ ] Create migration guide for users

---

## Summary

This design provides a complete, production-ready solution for asynchronous episode saving while maintaining backward compatibility and data integrity. The system is:

- **Thread-safe**: Uses locks to protect shared resources
- **Resilient**: Implements retry logic and error handling
- **Non-blocking**: Returns immediately to allow continuous data collection
- **Transparent**: Background saves happen without user intervention
- **Monitorable**: Provides status tracking and logging
- **Backward compatible**: Can fall back to synchronous saves if needed

The implementation eliminates the 15-30 second blocking time after each episode, improving data collection throughput by 2-3x and significantly enhancing the user experience.
