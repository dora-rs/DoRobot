# UX Improvements Design Document

This document describes two user experience improvements for the SO101 robot data collection system.

---

## Table of Contents
1. [Current Architecture Overview](#current-architecture-overview)
2. [Requirement 1: Single-Command Startup](#requirement-1-single-command-startup)
3. [Requirement 2: Combined Camera Visualization](#requirement-2-combined-camera-visualization)
4. [Implementation Plan](#implementation-plan)

---

## Current Architecture Overview

### Process Communication

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DORA Dataflow Process                              │
│  (dora run dora_teleoperate_dataflow.yml)                                   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ camera_top   │  │ camera_wrist │  │ arm_leader   │  │ arm_follower     │ │
│  │  (OpenCV)    │  │  (OpenCV)    │  │  (SO101)     │  │  (SO101)         │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
│         │                 │                 │                    │           │
│         └────────────┬────┴─────────────────┴────────────────────┘           │
│                      │                                                       │
│              ┌───────▼────────┐                                              │
│              │  dora_zeromq   │  ZeroMQ PAIR sockets (bind)                  │
│              │    (bridge)    │  - ipc:///tmp/dora-zeromq-so101-image        │
│              └───────┬────────┘  - ipc:///tmp/dora-zeromq-so101-joint        │
└──────────────────────┼──────────────────────────────────────────────────────┘
                       │ IPC
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                           CLI Process                                        │
│  (python operating_platform/core/main.py)                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        manipulator.py                                    ││
│  │  ZeroMQ PAIR sockets (connect)                                          ││
│  │  - recv_image_server() thread                                           ││
│  │  - recv_joint_server() thread                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                               │
│  ┌───────────────────────────▼─────────────────────────────────────────────┐│
│  │                         main.py                                          ││
│  │  - Daemon (robot control loop)                                          ││
│  │  - Record (data collection)                                             ││
│  │  - Camera visualization (cv2.imshow)                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

### Current Startup Process

**Manual 2-step process:**
1. Terminal 1: `bash scripts/run_so101_cli.sh` - Starts CLI, waits for ZeroMQ data (50s timeout)
2. Terminal 2: `bash scripts/run_so101_dora.sh` - Starts DORA dataflow, provides data

**Problems:**
- User must remember to start both processes
- User must use two terminals
- If user forgets to start DORA, CLI times out after 50s
- No graceful shutdown coordination

### Current Camera Visualization

```python
# main.py lines 274-277
for key in observation:
    if "image" in key:
        img = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Camera: {key}", img)
```

**Problems:**
- Creates separate window for each camera
- Windows overlap each other
- Windows may appear in random positions
- User must manually arrange windows

---

## Requirement 1: Single-Command Startup

### Design Goals
1. Single command to start both DORA and CLI
2. Proper startup order (DORA first, then CLI)
3. Readiness detection before starting CLI
4. Graceful shutdown of both processes
5. Clear status feedback to user

### Solution: Unified Launcher Script

Create `scripts/run_so101.sh` that manages both processes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_so101.sh (Launcher)                              │
│                                                                              │
│  1. Activate conda environment (dr-robot-so101)                             │
│  2. Start DORA dataflow in background                                       │
│  3. Wait for ZeroMQ sockets to be bound (readiness check)                   │
│  4. Activate conda environment (op)                                         │
│  5. Start CLI in foreground                                                 │
│  6. On exit: cleanup DORA process                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Readiness Detection Strategy

**Option A: File-based signal (Simple)**
- DORA writes a ready file when sockets are bound
- Launcher polls for file existence
- Pros: Simple, portable
- Cons: Requires DORA code modification

**Option B: Socket probe (Recommended)**
- Launcher probes ZeroMQ IPC socket files
- IPC sockets create files at `/tmp/dora-zeromq-so101-*`
- Pros: No DORA code modification needed
- Cons: Slightly more complex

**Option C: Process output parsing**
- Parse DORA stdout for "ready" message
- Pros: No file dependency
- Cons: Fragile, depends on log format

### Selected Approach: Option B (Socket Probe)

```bash
# Wait for ZeroMQ IPC socket files to exist
wait_for_sockets() {
    local timeout=30
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if [ -S "/tmp/dora-zeromq-so101-image" ] && [ -S "/tmp/dora-zeromq-so101-joint" ]; then
            echo "ZeroMQ sockets ready"
            return 0
        fi
        sleep 0.5
        elapsed=$((elapsed + 1))
    done
    echo "Timeout waiting for ZeroMQ sockets"
    return 1
}
```

### Process Management

```bash
# Trap signals for cleanup
cleanup() {
    echo "Shutting down..."
    if [ -n "$DORA_PID" ]; then
        kill $DORA_PID 2>/dev/null
        wait $DORA_PID 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM
```

### Implementation Files

| File | Purpose |
|------|---------|
| `scripts/run_so101.sh` | Unified launcher (new) |
| `scripts/run_so101_cli.sh` | CLI only (keep for debugging) |
| `scripts/run_so101_dora.sh` | DORA only (keep for debugging) |

---

## Requirement 2: Combined Camera Visualization

### Design Goals
1. Single window with all cameras
2. Left-right layout (horizontal concatenation)
3. Camera labels for identification
4. Consistent window position
5. Configurable layout (for future expansion)

### Solution: Camera Visualization Utility

Create a utility module that handles multi-camera display:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Combined Camera Window                                    │
│  ┌─────────────────────────────┬─────────────────────────────┐              │
│  │                             │                             │              │
│  │      Camera: image_top      │     Camera: image_wrist     │              │
│  │         (640x480)           │         (640x480)           │              │
│  │                             │                             │              │
│  │                             │                             │              │
│  │                             │                             │              │
│  └─────────────────────────────┴─────────────────────────────┘              │
│                        Total: 1280x480                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### API Design

```python
# operating_platform/utils/camera_display.py

class CameraDisplay:
    """Unified camera display utility for multi-camera visualization."""

    def __init__(self,
                 window_name: str = "Camera View",
                 layout: str = "horizontal",  # "horizontal" or "vertical" or "grid"
                 show_labels: bool = True,
                 label_height: int = 30):
        """
        Initialize camera display.

        Args:
            window_name: Name of the OpenCV window
            layout: How to arrange multiple cameras
            show_labels: Whether to show camera name labels
            label_height: Height of label bar in pixels
        """
        pass

    def show(self, images: dict[str, np.ndarray]) -> None:
        """
        Display multiple camera images in a single window.

        Args:
            images: Dict mapping camera name to image array (RGB format)
        """
        pass

    def close(self) -> None:
        """Close the display window."""
        pass
```

### Image Processing Pipeline

```python
def _combine_images(self, images: dict[str, np.ndarray]) -> np.ndarray:
    """Combine multiple images into single frame."""

    processed = []
    for name, img in sorted(images.items()):
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Add label if enabled
        if self.show_labels:
            labeled = self._add_label(bgr, name)
            processed.append(labeled)
        else:
            processed.append(bgr)

    # Concatenate based on layout
    if self.layout == "horizontal":
        return np.hstack(processed)
    elif self.layout == "vertical":
        return np.vstack(processed)
    else:  # grid
        return self._create_grid(processed)
```

### Integration Points

**main.py (recording loop):**
```python
# Before (lines 272-277):
if observation and not is_headless():
    for key in observation:
        if "image" in key:
            img = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Camera: {key}", img)

# After:
if observation and not is_headless():
    images = {k: v for k, v in observation.items() if "image" in k}
    camera_display.show(images)
```

**inference.py (similar changes needed)**

### Window Positioning

```python
def _setup_window(self):
    """Create and position the display window."""
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    # Position at top-left corner
    cv2.moveWindow(self.window_name, 50, 50)
```

### Implementation Files

| File | Purpose |
|------|---------|
| `operating_platform/utils/camera_display.py` | New camera display utility |
| `operating_platform/core/main.py` | Update visualization code |
| `operating_platform/core/inference.py` | Update visualization code |

---

## Implementation Plan

### Phase 1: Camera Display Utility
1. Create `camera_display.py` with `CameraDisplay` class
2. Implement horizontal concatenation
3. Add camera labels
4. Test standalone

### Phase 2: Integrate Camera Display
1. Update `main.py` recording loop
2. Update `main.py` reset view
3. Update `inference.py` (similar changes)
4. Test end-to-end

### Phase 3: Single-Command Startup
1. Create `run_so101.sh` launcher
2. Implement socket readiness check
3. Add process management
4. Test full workflow

### Phase 4: Documentation & Cleanup
1. Update CLAUDE.md with new commands
2. Update README if needed
3. Test on clean environment

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Different camera resolutions | Resize to match before concatenation |
| DORA startup slow | Increase timeout, add progress indicator |
| ZeroMQ socket cleanup issues | Delete stale socket files on startup |
| Different conda environments | Clear documentation, environment checks |

---

## Testing Checklist

- [ ] Single command starts both processes
- [ ] DORA starts before CLI connects
- [ ] CLI receives camera/joint data
- [ ] Combined camera window displays correctly
- [ ] Camera labels are visible
- [ ] Window doesn't overlap other windows
- [ ] Ctrl+C cleanly shuts down both processes
- [ ] 'e' key properly exits and cleans up
- [ ] Works on fresh terminal/environment
