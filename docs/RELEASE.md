# DoRobot Release Notes

This document tracks all changes made to the DoRobot data collection system.

---

## V23 (2025-11-28) - Unified Environment & UX Improvements

### Summary
Major UX overhaul with single-command startup, unified environment setup, combined camera visualization, and NPU support for Ascend hardware.

### Changes

**UX Improvement 1: Single-Command Startup**
- New unified launcher script `scripts/run_so101.sh` starts both DORA dataflow and CLI automatically
- Proper startup order: DORA first, then CLI after ZeroMQ sockets are ready
- Automatic cleanup of stale socket files from previous runs
- Graceful shutdown of both processes on exit (Ctrl+C or 'e' key)
- Configurable timeouts via environment variables

**UX Improvement 2: Combined Camera Visualization**
- New `CameraDisplay` class (`operating_platform/utils/camera_display.py`) combines all camera feeds into single window
- Horizontal layout with camera labels for easy identification
- Removes window clutter from multiple separate camera windows
- Consistent window positioning

**UX Improvement 3: Unified Environment Setup**
- New `scripts/setup_env.sh` creates single `dorobot` conda environment
- Supports multiple device types: CPU, CUDA 11.8/12.1/12.4, Ascend NPU
- Optional dependency groups: training, simulation, tensorflow, all
- Automatic installation of SO101 robot components and dependencies

**NPU Support (Ascend 310B)**
- Added torch-npu integration for Ascend AI processors
- CANN toolkit environment sourcing in launcher script
- Tested on Orange Pi AI Pro 20T development board

**File Changes:**
- New: `scripts/run_so101.sh` - Unified launcher
- New: `scripts/setup_env.sh` - Environment setup script
- New: `operating_platform/utils/camera_display.py` - Combined camera display
- New: `docs/DESIGN_UX_IMPROVEMENTS.md` - UX design document
- Modified: `operating_platform/robot/robots/so101_v1/manipulator.py` - NPU compatibility
- Modified: `pyproject.toml` - Optional dependency groups, NPU packages
- Modified: `README.md` - Updated documentation

---

## V22 (2025-11-28) - Timestamp & Logging Cleanup

### Changes
- Fixed timestamp calculation in `add_frame()` to always use frame_index/fps for consistency
- Added timestamp validation with monotonic increasing check
- Cleaned up verbose Chinese debug print statements in `remove_episode()`
- Replaced print() with logging.info()/warning() throughout dataset module

**File: `operating_platform/dataset/dorobot_dataset.py`**
- Line ~890: Calculate timestamp from frame_index, ignore frame dict timestamp
- Line ~895: Added validation for monotonically increasing timestamps
- Line ~1035: Cleaned up `remove_episode()` debug output

---

## V21 (2025-11-28) - Video Encoder Logging

### Changes
- Added progress logging for video encoding with timing info
- Skip message for already-encoded videos during resume

**File: `operating_platform/dataset/dorobot_dataset.py`**
- Line ~1219: Added `[VideoEncoder] Encoding N videos for episode X...` log
- Line ~1235: Added elapsed time logging after encoding complete

---

## V20 (2025-11-28) - Exit Sequence Fix

### Changes
- Fixed exit sequence to stop DORA daemon FIRST before waiting for async saves
- Prevents ARM hardware errors during save operations
- Async saver properly shutdown with `stop(wait_for_completion=True)`

**File: `operating_platform/core/main.py`**
- Line ~300: Stop daemon before saving (disconnect hardware gracefully)
- Line ~320: Use `async_saver.stop()` instead of just `wait_all_complete()`

---

## V19 (2025-11-28) - Recording Workflow Simplification

### Changes
- Merged 'n' (next episode) and 'p' (proceed after reset) key actions
- 'n' now saves immediately and starts new episode without reset prompt
- Removed reset timeout loop - continuous recording flow
- Added voice prompt: "Recording episode N" after save

**File: `operating_platform/core/main.py`**
- Line ~280: 'n' key now calls `record.save()` and immediately restarts
- Removed: Reset wait loop with 'p' key confirmation
- Removed: 60-second auto-proceed timeout

---

## V18 (2025-11-28) - Voice Prompts

### Changes
- Added voice prompts for recording state changes using existing `log_say()` function
- "Ready to start. Press N to save and start next episode."
- "Recording episode N."
- "End collection. Please wait for video encoding."

**File: `operating_platform/core/main.py`**
- Line ~270: Voice prompt on recording start
- Line ~295: Voice prompt on new episode
- Line ~305: Voice prompt on exit

---

## V17 (2025-11-28) - USB Port Cleanup & Hardware Handling

### Summary
Major reliability improvements for hardware disconnection and cleanup on exit.

### Problem 1: USB ports not released on exit
When the program exits (Ctrl+C, 'e' key, or crash), USB ports for cameras and robot arms remain locked, requiring physical reconnection.

### Problem 2: ARM errors during save
Robot arm communication errors occur during async save because hardware wasn't properly disconnected.

### Problem 3: ZeroMQ timeout spam
Console flooded with "Dora ZeroMQ Received Timeout" messages during normal polling.

### Changes

**File: `operating_platform/robot/components/camera_opencv/main.py`**
- Added signal handlers (SIGINT/SIGTERM) and atexit cleanup
- VideoCapture properly released on any exit path
- Global `_video_capture` reference for cleanup access

**File: `operating_platform/robot/components/arm_normal_so101_v1/main.py`**
- Added signal handlers and atexit cleanup for FeetechMotorsBus
- Disconnect with `disable_torque=True` on exit
- Proper cleanup on STOP event

**File: `operating_platform/robot/robots/so101_v1/dora_zeromq.py`**
- Added ZeroMQ socket and context cleanup on exit
- Signal handlers for graceful shutdown
- Removed timeout log messages (normal polling behavior)

**File: `operating_platform/robot/robots/aloha_v1_TODO/dora_zeromq.py`**
- Removed "Dora ZeroMQ Received Timeout" log message

**File: `operating_platform/robot/robots/pika_v1_TODO/manipulator.py`**
- Removed timeout log messages for Pika, VIVE, and Gripper receivers

**File: `operating_platform/core/daemon.py`**
- `stop()` now actually disconnects robot hardware
- Checks `robot.is_connected` before disconnect
- Proper error handling for disconnect failures

**File: `operating_platform/utils/video.py`**
- Converted print() to logging.info() for better visibility

---

## V16 (2025-11-27) - Shared Resource & Timeout Fixes

### Problem 1: Images not written for episodes after episode 0
After V15 fix, no errors appeared but:
- Episode 0 images exist (~1700 files)
- Episodes 1-9 have directories but 0 images
- `total_episodes: 0` - no saves completed
- Missing `data/` and `videos/` folders

### Root Cause 1
`save_episode()` unconditionally calls `stop_audio_writer()` at line 957.
When async worker processes episode 0's save on the SHARED dataset object,
it stops the audio_writer while the recording thread is still recording
episodes 1-9. This breaks the shared resource.

### Problem 2: `wait_all_complete()` timeout doesn't work
The timeout parameter was ineffective because `queue.join()` in Python
has no timeout parameter - the check only happens AFTER join() returns.

### Problem 3: Image write errors silently swallowed
`write_image()` just prints errors without proper logging, making debugging difficult.

### Changes

**File: `operating_platform/dataset/dorobot_dataset.py`**

1. Only stop audio writer in synchronous mode:
```python
# Line ~957-963
# IMPORTANT: Only stop audio writer in synchronous mode (episode_data is None)
# When called from async worker (episode_data provided), the recording thread
# is still recording and using the shared audio_writer. Stopping it here would
# break audio recording for subsequent episodes.
if not episode_data:
    self.stop_audio_writer()
    self.wait_audio_writer()
```

**File: `operating_platform/core/async_episode_saver.py`**

2. Fixed `wait_all_complete()` to use polling with actual timeout:
```python
# Line ~421-444
if timeout:
    # NOTE: queue.join() doesn't have a timeout parameter!
    # We use polling instead to implement proper timeout behavior.
    poll_interval = 0.5
    while True:
        with self._lock:
            pending = len(self._pending_saves)
            queue_size = self.save_queue.qsize()

        if pending == 0 and queue_size == 0:
            break

        elapsed = time.time() - start_time
        if elapsed > timeout:
            logging.warning(...)
            return False

        time.sleep(poll_interval)
```

**File: `operating_platform/dataset/image_writer.py`**

3. Improved error logging in `write_image()`:
```python
# Line ~71-84
def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path):
    import logging
    try:
        ...
    except Exception as e:
        # Log error with full traceback for debugging
        import traceback
        logging.error(f"[ImageWriter] Failed to write image {fpath}: {e}\n{traceback.format_exc()}")
```

---

## V15 (2025-11-27) - Race Condition Fix

### Problem
Episode saves occasionally fail with column length mismatch:
```
pyarrow.lib.ArrowInvalid: Column 2 named timestamp expected length 436 but got length 437
```

### Root Cause
Race condition between recording thread and save_async():
1. Recording thread (`process()`) continuously calls `add_frame()`
2. `save_async()` captures buffer reference
3. Before deep copy completes, recording thread adds another frame
4. Result: `size` counter doesn't match actual list lengths

### Changes

**File: `operating_platform/core/record.py`**

1. Added buffer lock in `__init__`:
```python
# Line ~133-135
# Lock to protect buffer swap during save_async (prevents race condition
# where recording thread adds frame while buffer is being captured)
self._buffer_lock = threading.Lock()
```

2. Use lock in `process()` around `add_frame()`:
```python
# Line ~225-227
# Use lock to prevent race condition with save_async buffer swap
with self._buffer_lock:
    self.dataset.add_frame(frame, self.record_cfg.single_task)
```

3. Use lock in `save_async()` for atomic buffer swap:
```python
# Line ~267-279
import copy

# CRITICAL: Use lock to atomically capture buffer and swap to new one
# This prevents the recording thread from adding frames during the swap
with self._buffer_lock:
    current_ep_idx = self.dataset.episode_buffer.get("episode_index", "?")
    logging.info(f"[Record] Queueing episode {current_ep_idx} for async save...")

    # Deep copy the buffer INSIDE the lock (before recording thread can add more frames)
    buffer_copy = copy.deepcopy(self.dataset.episode_buffer)

    # Create new episode buffer INSIDE the lock
    self.dataset.episode_buffer = self._create_new_episode_buffer()

# Queue save task with the copied buffer (outside lock to minimize lock hold time)
metadata = self.async_saver.queue_save(
    episode_buffer=buffer_copy,  # Pass the COPY, not the live buffer
    ...
)
```

---

## V14 (2025-11-27) - Dynamic Timeout Fix

### Problem
Long recordings (20+ seconds) fail because image write timeout (60s) is too short.
Log showed image writer taking 10+ minutes for longer recordings.

### Root Cause
Fixed 60 second timeout in `_wait_episode_images()` insufficient for episodes with many frames.

### Changes

**File: `operating_platform/dataset/dorobot_dataset.py`**

Changed `_wait_episode_images()` timeout from fixed 60s to dynamic calculation:
```python
# Line ~1154-1189
def _wait_episode_images(self, episode_index: int, episode_length: int, timeout_s: float | None = None) -> None:
    """
    Wait for a specific episode's images to be written.
    ...
    Args:
        ...
        timeout_s: Maximum time to wait in seconds. If None, calculates dynamically
                   based on episode length and number of cameras.
    """
    ...
    # Calculate dynamic timeout if not specified
    # Allow 0.5 seconds per image as a conservative estimate, with a minimum of 120 seconds
    # For a 20 second recording at 30fps with 2 cameras: 600 frames * 2 cameras * 0.5s = 600 seconds
    if timeout_s is None:
        num_images = episode_length * len(camera_keys)
        timeout_s = max(120.0, num_images * 0.5)
    ...
```

---

## V13 (2025-11-27) - Assertion & Image Writer Fixes

### Problem 1: Assertion Errors
```
AssertionError: len(video_files) == self.num_episodes * len(self.meta.video_keys)
```

### Root Cause 1
Global file count assertions fail with async save because:
- Episodes can be saved out of order
- Failed saves leave gaps in file counts
- Retries mean temporary inconsistencies

### Problem 2: Image File Not Found / Truncated
```
FileNotFoundError: [Errno 2] No such file or directory: '.../frame_000000.png'
OSError: image file is truncated
```

### Root Cause 2
Async saves started processing before image_writer finished writing all queued images.

### Changes

**File: `operating_platform/dataset/dorobot_dataset.py`**

1. Replaced global file count assertions with per-episode checks:
```python
# Line ~984-996 (REMOVED old assertions)
# OLD CODE (REMOVED):
# if len(self.meta.video_keys) > 0:
#     video_files = list(self.root.rglob("*.mp4"))
#     assert len(video_files) == self.num_episodes * len(self.meta.video_keys)
# parquet_files = list(self.root.rglob("*.parquet"))
# assert len(parquet_files) == self.num_episodes

# NEW CODE:
# NOTE: Removed file count assertions for async save compatibility.
# With async save, episodes may be saved out of order or have failed saves,
# so total file counts may not match num_episodes. Instead, we just verify
# that THIS episode's files were created successfully.
episode_parquet = self.root / self.meta.get_data_file_path(ep_index=episode_index)
if not episode_parquet.exists():
    raise RuntimeError(f"Failed to create parquet file for episode {episode_index}: {episode_parquet}")

if len(self.meta.video_keys) > 0:
    for key in self.meta.video_keys:
        episode_video = self.root / self.meta.get_video_file_path(episode_index, key)
        if not episode_video.exists():
            raise RuntimeError(f"Failed to create video file for episode {episode_index}: {episode_video}")
```

**File: `operating_platform/core/record.py`**

2. Added image_writer wait in `stop()`:
```python
# Line ~235-240
def stop(self):
    if self.running == True:
        self.running = False
        self.thread.join()
        self.dataset.stop_audio_writer()

    # CRITICAL: Wait for image_writer to finish ALL queued images BEFORE async saves
    # Without this, async saves will fail because images haven't been written yet
    if self.dataset.image_writer is not None:
        logging.info("[Record] Waiting for image_writer to complete all pending images...")
        self.dataset.image_writer.wait_until_done()
        logging.info("[Record] Image writer finished")
    ...
```

---

## V12 (2025-11-27) - Retry Failure Fix

### Problem
Retry attempts fail with:
```
KeyError: 'size' key not found in episode_buffer
```

### Root Cause
`save_episode()` uses `.pop()` to extract `size` and `task` from buffer:
```python
episode_length = episode_buffer.pop("size")  # Permanently removes key!
tasks = episode_buffer.pop("task")
```
When async saver retries a failed save, these keys are already gone.

### Changes

**File: `operating_platform/dataset/dorobot_dataset.py`**

Added deep copy at start of `save_episode()`:
```python
# Line ~910-934
def save_episode(self, episode_data: dict | None = None) -> int:
    import copy

    if episode_data:
        # IMPORTANT: Deep copy to preserve original buffer for retry compatibility.
        # The async saver may retry failed saves, and we use .pop() below which
        # modifies the buffer. Without this copy, retries would fail with
        # "size key not found in episode_buffer" because keys were already popped.
        episode_buffer = copy.deepcopy(episode_data)
    else:
        episode_buffer = self.episode_buffer

    validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

    # size and task are special cases that won't be added to hf_dataset
    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    ...
```

---

## Summary Table

| Version | Issue | File | Fix |
|---------|-------|------|-----|
| V12 | Retry fails (`size key not found`) | dorobot_dataset.py | Deep copy in save_episode() |
| V13 | Image not found / truncated | record.py | Wait for image_writer in stop() |
| V13 | Assertion errors (file counts) | dorobot_dataset.py | Per-episode file checks |
| V14 | Timeout for long recordings | dorobot_dataset.py | Dynamic timeout calculation |
| V15 | Race condition (column mismatch) | record.py | Lock for atomic buffer swap |
| V16 | Shared audio_writer stopped | dorobot_dataset.py | Only stop in sync mode |
| V16 | `wait_all_complete()` timeout broken | async_episode_saver.py | Use polling with timeout |
| V16 | Image write errors silent | image_writer.py | Proper logging |
| V17 | USB ports not released on exit | camera_opencv, arm_so101, dora_zeromq | Signal handlers + atexit cleanup |
| V17 | ARM errors during save | daemon.py | Proper robot disconnect |
| V17 | ZeroMQ timeout log spam | dora_zeromq.py, manipulator.py | Remove timeout log messages |
| V18 | No audio feedback | main.py | Voice prompts via log_say() |
| V19 | Reset prompt interrupts flow | main.py | Remove reset loop, 'n' saves directly |
| V20 | ARM errors on exit | main.py | Stop daemon FIRST before async saves |
| V21 | No encoding progress info | dorobot_dataset.py | Video encoder logging with timing |
| V22 | Timestamp sync errors | dorobot_dataset.py | Calculate from frame_index/fps |
| V22 | Verbose Chinese debug output | dorobot_dataset.py | Replace print() with logging |
| V23 | Two-step startup process | scripts/run_so101.sh | Single-command unified launcher |
| V23 | Multiple camera windows | camera_display.py | Combined camera visualization |
| V23 | Complex environment setup | scripts/setup_env.sh | Unified setup with device options |
| V23 | No NPU support | pyproject.toml, manipulator.py | Ascend NPU integration |

---

## Test Results

| Version | Episodes | Completed | Failed | Notes |
|---------|----------|-----------|--------|-------|
| V12 | 6 | 1 | 5 | Multiple error types |
| V13 | 6 | 1 | 5 | Still old assertions |
| V14 | 7 | 6 | 1 | Race condition on episode 5 |
| V15 | 10 | 0 | 10 | No errors but no saves (shared resource issue) |
| V16 | 10 | 10 | 0 | All episodes saved successfully |
| V17 | 10 | 10 | 0 | USB ports properly released |
| V18-V22 | - | - | - | Incremental improvements |
| V23 | 10 | 10 | 0 | Full workflow verified with unified launcher |

---

## Rollback Instructions

To rollback to a specific version, revert the changes listed for that version and all subsequent versions.

### Rollback V23 -> V22
1. Remove `scripts/run_so101.sh` and `scripts/setup_env.sh`
2. Remove `operating_platform/utils/camera_display.py`
3. Remove `docs/DESIGN_UX_IMPROVEMENTS.md`
4. Revert `main.py` camera display changes (restore individual `cv2.imshow()` per camera)
5. Revert `pyproject.toml` optional dependency groups and NPU packages

### Rollback V22 -> V21
1. In `dorobot_dataset.py`, revert timestamp calculation to use frame dict timestamp
2. Restore Chinese debug print statements in `remove_episode()`

### Rollback V21 -> V20
1. In `dorobot_dataset.py`, remove video encoder timing/logging messages

### Rollback V20 -> V19
1. In `main.py`, move `daemon.stop()` back after async saves complete
2. Use `wait_all_complete()` instead of `async_saver.stop()`

### Rollback V19 -> V18
1. In `main.py`, restore reset wait loop with 'p' key confirmation
2. Restore 60-second auto-proceed timeout
3. Separate 'n' key behavior (don't save, just end episode)

### Rollback V18 -> V17
1. Remove `log_say()` voice prompt calls from `main.py`

### Rollback V17 -> V16
1. Remove signal handlers and atexit cleanup from `camera_opencv/main.py`
2. Remove signal handlers and atexit cleanup from `arm_normal_so101_v1/main.py`
3. Remove signal handlers and cleanup from `so101_v1/dora_zeromq.py`
4. Restore ZeroMQ timeout log messages in `dora_zeromq.py`, `manipulator.py`
5. In `daemon.py`, revert `stop()` to empty pass statement
6. In `video.py`, change `logging.info()` back to `print()`

### Rollback V16 -> V15
1. In `dorobot_dataset.py`, remove the `if not episode_data:` condition around `stop_audio_writer()`/`wait_audio_writer()`
2. In `async_episode_saver.py`, restore `queue.join()` in `wait_all_complete()` instead of polling
3. In `image_writer.py`, change `logging.error()` back to `print()`

### Rollback V15 -> V14
Remove the `_buffer_lock` and associated `with self._buffer_lock:` blocks from `record.py`.

### Rollback V14 -> V13
Change `_wait_episode_images()` timeout back to `timeout_s: float = 60.0` and remove the dynamic calculation.

### Rollback V13 -> V12
1. Restore global file count assertions in `dorobot_dataset.py`
2. Remove `image_writer.wait_until_done()` from `record.py` stop()

### Rollback V12 -> Original
Remove `copy.deepcopy()` from `save_episode()`.
