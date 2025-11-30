# Cloud Offload Feasibility Study

## Problem Statement

After pressing 'e' to finish data collection, the system spends ~1 hour on:
1. Writing PNG images from memory buffers to disk
2. Encoding PNG images to MP4 videos (ffmpeg)

For 10 episodes with 2 cameras, this is extremely slow on Orange Pi 310B hardware.

**Key Insight**: The data will be uploaded to cloud for training anyway. Why not offload the slow processing to cloud?

---

## Current Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL (Orange Pi 310B)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Recording Loop                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  add_frame() → image_writer queue → PNG files (images/)             │   │
│  │             → episode_buffer (action, state, timestamp)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  User presses 'e'                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. _wait_episode_images() - wait for PNG files (SLOW: minutes)      │   │
│  │  2. _save_episode_table() - write parquet (fast)                     │   │
│  │  3. encode_episode_videos() - PNG→MP4 (VERY SLOW: 1 hour for 10 ep) │   │
│  │  4. Delete PNG files                                                 │   │
│  │  5. Save metadata                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Final Output:                                                              │
│  ├── data/chunk-000/episode_XXXXXX.parquet                                 │
│  ├── videos/chunk-000/{camera_key}/episode_XXXXXX.mp4                      │
│  └── meta/*.json                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Upload to cloud
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLOUD (Training Server)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  train.py loads dataset from videos/                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Cloud Offload Architecture

### Option A: Upload Raw PNG + Metadata (Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL (Orange Pi 310B)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Recording Loop (unchanged)                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  add_frame() → image_writer queue → PNG files (images/)             │   │
│  │             → episode_buffer (action, state, timestamp)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  User presses 'e' (NEW: Fast path)                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. _wait_episode_images() - wait for PNG files                      │   │
│  │  2. _save_episode_table() - write parquet                            │   │
│  │  3. Package & upload to cloud:                                       │   │
│  │     - images/{camera}/episode_XXXXXX/*.png (compressed tar)          │   │
│  │     - data/chunk-000/episode_XXXXXX.parquet                          │   │
│  │     - meta/*.json                                                    │   │
│  │  4. Mark episode as "pending_encode" in metadata                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Upload Data:                                                               │
│  ├── images/{camera}/episode_XXXXXX/*.png  (~500MB for 10 ep, 2 cameras)   │
│  ├── data/chunk-000/episode_XXXXXX.parquet (~50KB per episode)             │
│  └── meta/*.json                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Upload (rsync/scp/S3)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLOUD (Processing Server)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Cloud Encoder Service (runs automatically on upload trigger)               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. Receive upload notification (S3 trigger / webhook / cron)        │   │
│  │  2. For each pending episode:                                        │   │
│  │     - ffmpeg PNG→MP4 (fast on cloud GPU/CPU)                         │   │
│  │     - Update metadata: "pending_encode" → "ready"                    │   │
│  │     - Delete PNG files                                               │   │
│  │  3. Notify local device (optional)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Final Output (same format as current):                                     │
│  ├── data/chunk-000/episode_XXXXXX.parquet                                 │
│  ├── videos/chunk-000/{camera_key}/episode_XXXXXX.mp4                      │
│  └── meta/*.json                                                           │
│                                                                             │
│  Training Server can now use the dataset                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Size Analysis

Based on example dataset `/Users/nupylot/Public/so101-test-1127ok`:

| Item | Size | Notes |
|------|------|-------|
| **3 episodes** | | |
| videos/ | 17MB | MP4 encoded (6 videos) |
| data/ | 24KB | Parquet files |
| images/ | 8KB | Empty (deleted after encoding) |
| meta/ | ~20KB | JSON metadata |
| **Total** | ~17MB | |

**Extrapolation for 10 episodes:**

| Item | Estimated Size | Notes |
|------|----------------|-------|
| Raw PNG images | ~500MB | 10 ep × 2 cameras × 300 frames × 80KB/frame |
| MP4 videos | ~60MB | ~6MB per video × 10 ep × 2 cameras |
| Parquet data | ~100KB | ~10KB per episode |

**Key Observation**:
- Raw PNG is ~8x larger than encoded MP4
- But upload 500MB is faster than encoding 1 hour locally

---

## Upload Time Estimation

| Network Speed | 500MB Upload Time |
|---------------|-------------------|
| 10 Mbps | 6-7 minutes |
| 50 Mbps | 1-2 minutes |
| 100 Mbps | ~40 seconds |
| 1 Gbps | ~4 seconds |

**Comparison**:
- Local encoding: ~1 hour
- Cloud upload + cloud encode: ~10 minutes total (worst case 10 Mbps)

---

## Implementation Options

### Option A: Simple rsync/scp Upload (Recommended for MVP)

**Pros:**
- Minimal code changes
- Works with any Linux server
- No cloud vendor lock-in

**Implementation:**
```python
# In main.py after 'e' key pressed
def offload_to_cloud(dataset_root: Path, cloud_host: str, cloud_path: str):
    """Upload raw data to cloud for encoding."""
    # 1. Create tar of images (compress on-the-fly)
    tar_cmd = f"tar -czf - -C {dataset_root} images/ data/ meta/"

    # 2. Stream to cloud via ssh
    upload_cmd = f"{tar_cmd} | ssh {cloud_host} 'tar -xzf - -C {cloud_path}'"

    # 3. Trigger encoding on cloud
    encode_cmd = f"ssh {cloud_host} 'python encode_dataset.py {cloud_path}'"
```

**Cloud side script (encode_dataset.py):**
```python
def encode_pending_episodes(dataset_root: Path):
    """Encode PNG to MP4 for all pending episodes."""
    for episode_dir in (dataset_root / "images").glob("*/episode_*"):
        episode_idx = int(episode_dir.name.split("_")[1])
        camera_key = episode_dir.parent.name

        video_path = dataset_root / f"videos/chunk-000/{camera_key}/episode_{episode_idx:06d}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # Fast encoding on cloud (GPU or fast CPU)
        encode_video_frames(episode_dir, video_path, fps=30)

        # Delete PNG after successful encode
        shutil.rmtree(episode_dir)
```

### Option B: S3/Cloud Storage with Lambda Trigger

**Pros:**
- Serverless, auto-scaling
- Pay per use
- Built-in redundancy

**Cons:**
- Cloud vendor lock-in
- More complex setup
- Costs can accumulate

### Option C: Streaming Upload During Recording

**Pros:**
- Zero wait time at end
- Continuous backup

**Cons:**
- Network bandwidth during recording may affect performance
- Complex implementation
- Need reliable network

---

## Recommended Approach

### Phase 1: MVP (Option A with rsync)

1. **Local changes:**
   - Add `--cloud-offload` flag to main.py
   - Skip video encoding if flag is set
   - Upload images/ + data/ + meta/ via rsync/scp
   - Keep images/ locally until upload confirmed

2. **Cloud script:**
   - Simple Python script to encode pending episodes
   - Run via cron or triggered by upload notification
   - Same `encode_video_frames()` function (portable)

3. **Data format unchanged:**
   - Final dataset structure identical to current
   - Training pipeline requires no changes

### Phase 2: Optimization

1. **Parallel upload during recording:**
   - Upload completed episodes while recording continues
   - Use background thread for upload

2. **Compression optimization:**
   - Use `tar.zst` (Zstandard) for better compression ratio
   - Or upload PNG directly to S3 (parallel multipart)

3. **Progress tracking:**
   - Local metadata tracks upload status
   - Cloud confirms encoding status
   - Resume capability for interrupted uploads

---

## Time Comparison

| Step | Current (Local) | Cloud Offload |
|------|-----------------|---------------|
| Image writing | ~5 min | ~5 min (same) |
| Video encoding | ~55 min | 0 (skipped) |
| Upload | 0 | ~5 min (50 Mbps) |
| Cloud encoding | 0 | ~2 min (fast CPU/GPU) |
| **Total** | **~60 min** | **~12 min** |

**Speedup: ~5x faster**

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Network failure during upload | Keep local copy until confirmed; resume capability |
| Cloud encoding failure | Retry logic; fallback to local encoding |
| Data format incompatibility | Use same encode_video_frames() function |
| Cloud storage costs | Compress before upload; delete after training |
| Security (data in transit) | Use SSH/HTTPS; encrypt sensitive data |

---

## Decision Matrix

| Criteria | Local (Current) | Cloud Offload |
|----------|-----------------|---------------|
| Time to complete | 1 hour | 10-15 min |
| Network required | No | Yes (upload phase) |
| Cloud compute required | No | Yes (but fast) |
| Code changes | None | Moderate |
| Data format changes | None | None |
| Rollback option | N/A | Yes (can still encode locally) |

---

## Conclusion

**Recommendation: Implement Cloud Offload (Option A)**

The cloud offload approach is feasible and provides significant time savings:
- **5x faster** end-to-end time
- **Same data format** - no training pipeline changes
- **Simple implementation** - rsync + Python script
- **Low risk** - can fall back to local encoding

**Next Steps:**
1. Implement `--cloud-offload` flag in main.py
2. Create cloud-side encode_dataset.py script
3. Test with 10-episode collection
4. Measure actual time savings

---

## Data-Platform Integration Analysis

### Current Training Workflow

The data-platform uses `upload_train_rsync.py` to orchestrate the full workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        upload_train_rsync.py (Local)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. login() → Get auth token                                                │
│  2. start_transaction() → Get SSH credentials & transaction ID              │
│  3. rsync_data() → Upload to /root/{s3_data_path}/                          │
│  4. mark_upload_complete() → POST /transactions/{id}/upload-complete        │
│  5. poll_transaction_status() / SSHTrainingMonitor → Wait for training      │
│  6. rsync_download_folder() → Download trained model                        │
│  7. modify_config_device() → Change "npu" → "cuda" in config.json           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ API Call
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            api.py (Cloud Server)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  POST /transactions/{id}/upload-complete                                    │
│  ├── Spawns thread: start_training_workflow(transaction_id)                 │
│  └── Updates transaction status to "TRAINING"                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ LangGraph Workflow
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           chat1.py (Training Workflow)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. check_ec2_instance → Verify training server available                   │
│  2. run_training → SSH to server, execute train.py                          │
│  3. monitor_training → Poll training progress                               │
│  4. Complete → Update transaction status                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### meta/info.json Context Requirements

The `meta/info.json` contains video metadata that must be populated during encoding:

```json
{
  "features": {
    "observation.images.image_top": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "info": {
        "video.fps": 30.0,
        "video.height": 480,
        "video.width": 640,
        "video.channels": 3,
        "video.codec": "h264",          // ← Set by ffprobe after encoding
        "video.pix_fmt": "yuvj420p",    // ← Set by ffprobe after encoding
        "video.is_depth_map": false,
        "has_audio": false
      }
    }
  }
}
```

**Key Insight**: The `video.codec` and `video.pix_fmt` fields are populated by `get_video_info()` in `video.py` after encoding. This function uses ffprobe to read the actual video file properties.

---

## Encoding Integration Options

### Option 1: Pre-processing Task Before Training (Recommended)

Insert encoding as a new step in the LangGraph workflow, before training starts.

```
upload_train_rsync.py                    api.py / chat1.py
┌─────────────────────┐                  ┌─────────────────────┐
│ Upload raw PNG      │ ──rsync──────────│ Receive data        │
│ + parquet + meta    │                  │                     │
│                     │                  │ NEW: encode_videos()│
│                     │                  │ - ffmpeg PNG→MP4    │
│                     │                  │ - Update info.json  │
│                     │                  │ - Delete PNG files  │
│                     │                  │                     │
│                     │                  │ run_training()      │
│                     │                  │ - train.py on data  │
│                     │                  │                     │
│ Download model ────────rsync───────────│ Return model        │
└─────────────────────┘                  └─────────────────────┘
```

**Implementation in chat1.py:**

```python
# Add new node to LangGraph workflow
def encode_videos(state):
    """Pre-process: encode PNG images to MP4 videos"""
    data_path = state["s3_data_path"]
    remote_path = f"/root/{data_path}"

    # SSH to training server and run encoding
    ssh_cmd = f'''
    cd {remote_path}
    python encode_dataset.py .
    '''

    # encode_dataset.py will:
    # 1. Find all PNG images in images/
    # 2. Encode to MP4 in videos/
    # 3. Update meta/info.json with video codec info
    # 4. Delete PNG files

    return {"current_step": "encoding_completed"}

# Modify workflow graph
workflow.add_node("encode_videos", encode_videos)
workflow.add_edge("check_ec2_instance", "encode_videos")
workflow.add_edge("encode_videos", "run_training")
```

**Pros:**
- Clean separation of concerns (encoding vs training)
- Encoding failure doesn't affect training retry logic
- Easy to monitor/debug each phase separately
- Can reuse existing encode_video_frames() function
- info.json update happens on same server that has the videos

**Cons:**
- Need to modify LangGraph workflow in chat1.py
- Adds a new state to track

---

### Option 2: Encoding in train.py via SSH

Let train.py detect unencoded data and encode before training.

```python
# In train.py or a wrapper script
def ensure_videos_encoded(dataset_root: Path):
    """Check if encoding needed, encode if so"""
    images_dir = dataset_root / "images"

    if images_dir.exists() and any(images_dir.iterdir()):
        logging.info("Found raw PNG images, encoding to MP4...")

        # Run encoding
        encode_pending_episodes(dataset_root)

        # Update info.json with video metadata
        update_video_info_metadata(dataset_root)

    logging.info("Videos ready for training")
```

**Pros:**
- No changes to data-platform workflow
- Self-contained - train.py handles everything
- Works even if called manually without data-platform

**Cons:**
- Mixes encoding concern with training
- Training script becomes more complex
- If encoding fails, need to restart entire training process
- Encoding time included in "training time" metrics (misleading)

---

### Option 3: Separate Encoding Service (Cloud-side Daemon)

Run a daemon on the training server that watches for new uploads.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       Training Server (Cloud)                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  encode_daemon.py (always running)                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  while True:                                                        │  │
│  │      for dataset in find_pending_datasets("/root/"):                │  │
│  │          if has_unencoded_images(dataset):                          │  │
│  │              encode_videos(dataset)                                 │  │
│  │              update_info_json(dataset)                              │  │
│  │              mark_ready(dataset)                                    │  │
│  │      sleep(10)                                                      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  train.py (triggered by workflow)                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  wait_for_encoding_complete(dataset_path)                           │  │
│  │  train_on_dataset(dataset_path)                                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Decoupled from both upload and training
- Can start encoding as soon as data arrives
- Parallel encoding while training previous dataset

**Cons:**
- Additional process to manage
- Need coordination between daemon and training
- More complex deployment

---

## Decision Matrix: Integration Options

| Criteria | Option 1: Pre-processing | Option 2: In train.py | Option 3: Daemon |
|----------|-------------------------|----------------------|------------------|
| Code changes | Moderate (chat1.py) | Minimal (train.py) | High (new daemon) |
| Separation of concerns | Excellent | Poor | Excellent |
| Failure isolation | Excellent | Poor | Excellent |
| Debugging ease | Good | Moderate | Complex |
| Deployment complexity | Low | Low | High |
| Training time accuracy | Accurate | Inflated | Accurate |
| info.json update | Straightforward | Straightforward | Straightforward |

---

## Recommended Solution: Option 1 (Pre-processing Task)

**Rationale:**

1. **Clean Architecture**: Encoding is a data preparation step, not training. It belongs before training in the workflow.

2. **Failure Handling**: If encoding fails, the workflow can retry just the encoding step without re-uploading data or restarting training.

3. **Metrics Accuracy**: Training time metrics will accurately reflect training duration, not encoding + training.

4. **Minimal Changes**: Only requires adding one new node to the existing LangGraph workflow.

5. **info.json Compatibility**: The encoding script runs on the same server where the data resides, so it can easily update `meta/info.json` with accurate video metadata from ffprobe.

**Implementation Plan:**

1. **Create `encode_dataset.py` script** (place in data-platform or on training server):
   ```python
   def main(dataset_root: Path):
       # 1. Find all PNG images
       # 2. For each episode:
       #    - Encode PNG → MP4 using encode_video_frames()
       #    - Get video info using ffprobe
       # 3. Update meta/info.json with video.codec, video.pix_fmt
       # 4. Delete PNG files after successful encoding
   ```

2. **Modify chat1.py workflow**:
   - Add `encode_videos` node
   - Insert between `check_ec2_instance` and `run_training`

3. **Modify local data collection**:
   - Skip video encoding when `--cloud-offload` flag is set
   - Upload images/ + data/ + meta/ (without video info in features)

4. **Update upload_train_rsync.py** (optional):
   - Add status message for encoding phase
   - Monitor encoding progress in addition to training progress

---

## info.json Update Strategy

The cloud encoder must update `meta/info.json` after encoding:

```python
def update_info_json_video_metadata(dataset_root: Path):
    """Update info.json with actual video codec info after encoding."""
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for key, feature in info["features"].items():
        if feature.get("dtype") == "video":
            # Find corresponding video file
            video_path = find_first_video_for_feature(dataset_root, key)
            if video_path:
                video_info = get_video_info(video_path)
                feature["info"] = video_info

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
```

This ensures the training pipeline sees the correct video codec information when loading the dataset.

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| Integration point | Pre-processing task in LangGraph workflow |
| Encoding location | Cloud training server |
| info.json update | Done by cloud encoder after ffmpeg |
| Local changes | Skip encoding, upload raw PNG |
| Workflow changes | Add `encode_videos` node in chat1.py |
| Fallback | Can still encode locally if cloud fails |

---

## Implementation Status

**Date Implemented**: November 2024

### Changes Made

#### 1. Data-Platform: `encode_dataset.py` (New File)

Location: `/Users/nupylot/Public/data-platform/encode_dataset.py`

A standalone cloud-side encoder script that:
- Finds all unencoded episodes (those with `images/` but no `videos/`)
- Encodes PNG images to MP4 using ffmpeg with libx264
- Updates `meta/info.json` with video codec information from ffprobe
- Deletes PNG files after successful encoding

Usage:
```bash
python encode_dataset.py /path/to/dataset --preset fast --crf 23
```

#### 2. Data-Platform: `upload_train_rsync.py` Modifications

Added `--cloud-offload` CLI flag:
- Passes `cloud_offload: true` in training_config to the API
- Logs cloud offload mode status

Modified functions:
- `start_transaction()`: Now accepts `cloud_offload` parameter and includes it in training config
- `main()`: Added `--cloud-offload` argument parsing

#### 3. Data-Platform: `chat1.py` LangGraph Workflow

Added new `encode_videos` node in the workflow:
- Inserted between `copy_data_to_ec2` and `run_training`
- Only executes encoding when `cloud_offload: true` in training_config
- Runs `encode_dataset.py` via SSH on the cloud server
- Updates workflow routing to include the new step

Modified:
- Added `encode_videos()` function (~100 lines)
- Added `encode_videos` to workflow graph nodes
- Updated `copy_data_to_ec2` to route to `encode_videos` instead of `run_training`
- Updated `workflow_router` valid_nodes list

#### 4. DoRobot: `dorobot_dataset.py` Modifications

Added `skip_encoding` parameter to `save_episode()` method:
- When `skip_encoding=True`:
  - Skips video encoding (no `encode_episode_videos()` call)
  - Keeps PNG images (no cleanup of `images/` directory)
  - Skips video file existence check
- When `skip_encoding=False` (default): Behavior unchanged (backward compatible)

### Backward Compatibility

All changes maintain backward compatibility:

1. **Default Behavior**: Without `--cloud-offload` flag, the system works exactly as before
2. **API Compatibility**: The `/transactions/start` endpoint accepts the existing request format
3. **Dataset Format**: Final dataset structure is identical after cloud encoding
4. **Training Pipeline**: No changes required to training scripts

### Usage

#### Standard Mode (Local Encoding)
```bash
# On edge device - encodes videos locally
python train.py --input /path/to/dataset --output /path/to/output

# Via upload script
python upload_train_rsync.py --input /path/to/dataset --output /path/to/output
```

#### Cloud Offload Mode (for slow edge devices)
```bash
# On edge device - skips encoding, uploads raw PNG
python upload_train_rsync.py --input /path/to/dataset --output /path/to/output --cloud-offload
```

#### Direct Dataset Save with Skip Encoding
```python
# In data collection code
dataset.save_episode(skip_encoding=True)  # Keep PNG for cloud encoding
```

### Testing Checklist

- [ ] Standard mode (without `--cloud-offload`) works unchanged
- [ ] Cloud offload mode uploads images/ directory
- [ ] `encode_videos` node executes on cloud server
- [ ] `encode_dataset.py` successfully encodes PNG to MP4
- [ ] `meta/info.json` contains correct video codec info after encoding
- [ ] Training runs successfully after cloud encoding
- [ ] Trained model downloads correctly
