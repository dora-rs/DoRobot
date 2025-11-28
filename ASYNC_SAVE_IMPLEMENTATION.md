# Async Episode Save Implementation - Summary

## Implementation Completed

The asynchronous episode save system has been successfully implemented based on the design in `design.md`. This eliminates the 15-30 second blocking delay when saving episodes during data collection.

## Changes Made

### 1. New File: `operating_platform/core/async_episode_saver.py`
Created the core `AsyncEpisodeSaver` class that:
- Queues episode save operations in background thread
- Returns metadata immediately (<100ms) without blocking
- Handles image writing, video encoding, and metadata updates asynchronously
- Provides retry logic and error handling
- Tracks save status (pending/completed/failed)

**Key Classes:**
- `EpisodeSaveTask`: Represents a single save operation
- `EpisodeMetadata`: Immediate return value with episode index and queue position
- `AsyncEpisodeSaver`: Main async save coordinator

### 2. Thread Safety: `operating_platform/dataset/dorobot_dataset.py`
Added thread-safe locks to `DoRobotDatasetMetadata`:
- Added `import threading` at module level
- Added `self._meta_lock = threading.Lock()` in `__init__` and `create()` methods
- Protected `save_episode()` method with lock (line 288-313)
- Protected `add_task()` method with lock (line 262-275)

**Critical for data integrity:** Multiple background threads can now safely update metadata counters without race conditions.

### 3. Record Class Updates: `operating_platform/core/record.py`
Modified `Record` class to support async saves:

**Added:**
- `from operating_platform.core.async_episode_saver import AsyncEpisodeSaver, EpisodeMetadata`
- `self.use_async_save` flag (defaults to True)
- `self.async_saver` instance in `__init__` (line 123-128)
- `save_async()` method - queues episode and returns immediately (line 213-236)
- `save_sync()` method - original synchronous save (line 238-277)
- Modified `save()` - routes to async or sync based on config (line 279-286)

**Modified:**
- `start()` - starts async saver background thread (line 171-175)
- `stop()` - waits for pending saves before stopping (line 203-208)

### 4. Configuration: `operating_platform/core/record.py` (RecordConfig)
Added async save configuration options (line 106-114):
```python
# Enable asynchronous episode saving (non-blocking save operations)
use_async_save: bool = True
# Maximum number of episodes that can be queued for async saving
async_save_queue_size: int = 10
# Timeout in seconds for save operations
async_save_timeout_s: int = 300
# Maximum retry attempts for failed saves
async_save_max_retries: int = 3
```

### 5. Main Recording Loop: `operating_platform/core/main.py`
Updated recording loop to handle async saves (line 300-333):

**On 'n' key (next episode):**
- Calls `record.save()` which returns immediately
- Logs episode_index and queue position
- Checks for failed saves from previous episodes

**On 'e' key (exit):**
- Calls `record.save()` to queue final episode
- Waits for all pending saves to complete before exiting
- Logs final statistics (queued/completed/failed)

## Data Format Compatibility

**✓ GUARANTEED:** The async implementation uses the **exact same** `dataset.save_episode()` method as the synchronous version. It just calls it from a background thread. This ensures:
- Same parquet file format
- Same video encoding
- Same metadata structure
- Same JSONL files
- Compatible with existing training pipeline

Reference data structure validated against: `/Users/nupylot/Public/so101-test-1126-ok`

## How It Works

### Before (Synchronous):
```
User presses 'n'
  → record.stop() (fast)
  → record.save() blocks for 15-30 seconds:
      - wait for image writer (slow)
      - encode videos (slow)
      - save metadata (fast)
  → User can finally press 'p' to continue
```

### After (Asynchronous):
```
User presses 'n'
  → record.stop() (fast)
  → record.save() returns in <100ms:
      - allocates episode_index
      - deep copies episode_buffer
      - queues save task
      - returns metadata immediately
  → User can immediately press 'p' to continue

Background thread processes queue:
  → waits for image writer
  → encodes videos
  → saves metadata
  → marks episode as completed
```

## Configuration Options

### Enable/Disable Async Save
```bash
# Enable (default)
python operating_platform/core/main.py --record.use_async_save=true

# Disable (fallback to synchronous)
python operating_platform/core/main.py --record.use_async_save=false
```

### Adjust Queue Size
```bash
# Default: 10 episodes
python operating_platform/core/main.py --record.async_save_queue_size=20
```

### Adjust Retry Settings
```bash
python operating_platform/core/main.py \
  --record.async_save_max_retries=5 \
  --record.async_save_timeout_s=600
```

## Monitoring Save Status

The async saver logs detailed status information:

**On save:**
```
[AsyncEpisodeSaver] ✓ Queued episode 5 (queue_pos=2, frames=87, elapsed=23.4ms)
```

**During background save:**
```
[AsyncEpisodeSaver] Starting save for episode 5
[AsyncEpisodeSaver] ✓ Episode 5 saved successfully in 18.32s (frames=87)
```

**On exit:**
```
[Record] Waiting for 3 pending saves...
[AsyncEpisodeSaver] All saves completed in 54.21s
Save stats: queued=10 completed=10 failed=0
```

## Error Handling

### Queue Full
If more than `async_save_queue_size` episodes are queued:
- New saves fail immediately
- Error logged: `Save queue full! Episode X dropped`
- Previous episodes continue processing

### Save Failure
If save fails (disk full, corruption, etc.):
- Automatic retry with exponential backoff (2^retry_count seconds)
- Max retries: `async_save_max_retries` (default 3)
- After max retries, episode marked as failed
- Failure logged with episode index

### Disk Full
```
[AsyncEpisodeSaver] DISK FULL! Stopping recording
```
Triggers emergency stop to prevent data loss.

## Testing Checklist

Before deploying to production, verify:

- [ ] **Single Episode**: Record 1 episode, verify data format matches reference
- [ ] **Rapid Episodes**: Press 'n' 5 times quickly, verify all save correctly
- [ ] **Exit During Save**: Press 'e' while saves pending, verify all complete
- [ ] **Training Compatibility**: Train a policy on async-saved data, verify no errors
- [ ] **Sync Fallback**: Set `use_async_save=false`, verify synchronous save still works
- [ ] **Queue Overflow**: Queue > 10 episodes, verify error handling
- [ ] **Disk Space**: Fill disk during save, verify graceful failure

## Performance Improvements

### Expected Results:
- **Save time**: 15-30s → <0.1s (150-300x faster)
- **Episode throughput**: 2-3 eps/min → 6-10 eps/min (2-3x increase)
- **User experience**: No blocking delays, immediate response

### Actual Performance:
Test with your hardware and report:
```bash
# Run recording session, count episodes collected in 10 minutes
# Before async: ~20-30 episodes
# After async: ~60-100 episodes (expected)
```

## Rollback Plan

If issues occur:
1. Set `--record.use_async_save=false` in command line
2. Fallback to synchronous save (original behavior)
3. No code changes needed
4. No data loss

## Future Enhancements

Potential improvements (not yet implemented):
1. **Priority Queue**: Save recent episodes first
2. **Cloud Upload**: Auto-upload to S3/HuggingFace in background
3. **Progress Bar**: Real-time save progress display
4. **Compression**: Background video compression
5. **Checkpoint Recovery**: Resume failed saves after crash

## Files Modified

1. **NEW**: `operating_platform/core/async_episode_saver.py` (complete file, 550 lines)
2. `operating_platform/dataset/dorobot_dataset.py`: Added `import threading`, added `_meta_lock` to protect metadata
3. `operating_platform/core/record.py`: Added async save methods, modified `Record` class
4. `operating_platform/core/main.py`: Updated recording loop for async saves

## Migration Guide

### For Existing Users:
1. Pull latest code
2. No changes needed - async save is enabled by default
3. First run will be with async save
4. If issues occur, add `--record.use_async_save=false`

### For New Users:
- Async save is default behavior
- Enjoy immediate response time!

## Code Quality

- ✓ Thread-safe (locks on shared state)
- ✓ Error handling (retry logic, timeouts)
- ✓ Logging (detailed status tracking)
- ✓ Backward compatible (sync fallback available)
- ✓ Data format guaranteed (uses same save_episode())
- ✓ Well documented (docstrings, comments)

## Support

For issues or questions:
1. Check logs for `[AsyncEpisodeSaver]` messages
2. Try synchronous fallback: `--record.use_async_save=false`
3. Report issue with:
   - Log output
   - Failed episode index
   - Disk space available
   - Number of pending saves

## Summary

The async episode save system has been fully implemented and is ready for testing. It provides:
- **Non-blocking saves**: Return in <100ms instead of 15-30s
- **Thread safety**: Locks protect metadata from race conditions
- **Error resilience**: Retry logic and error tracking
- **Data compatibility**: Uses same save method, guaranteed format compatibility
- **Monitoring**: Detailed logging and status tracking
- **Backward compatibility**: Can fallback to synchronous saves

**Next Step:** Test with actual robot hardware to validate performance and data format.
