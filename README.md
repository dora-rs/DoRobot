# DoRobot-Test

> Dora LeRobot Version - A robotics operating platform for robot control, data collection, and policy training.

## Quick Start

### Get the Project

```bash
git clone https://github.com/ustbxcj/DoRobotTest.git
cd DoRobotTest
```

### Automated Environment Setup (Recommended)

Use the setup script to create a unified conda environment with all dependencies:

```bash
# Core only - for data collection (fastest install)
bash scripts/setup_env.sh

# With training dependencies (for policy training)
bash scripts/setup_env.sh --training

# With CUDA support
bash scripts/setup_env.sh --cuda 12.4

# With CUDA + training
bash scripts/setup_env.sh --cuda 12.4 --training

# With Ascend NPU support (310B)
bash scripts/setup_env.sh --npu

# NPU + training
bash scripts/setup_env.sh --npu --training

# All dependencies
bash scripts/setup_env.sh --all
```

**Setup Options:**

| Option | Description |
|--------|-------------|
| `--name NAME` | Environment name (default: dorobot) |
| `--python VER` | Python version (default: 3.11) |
| `--device DEVICE` | Device: cpu, cuda11.8, cuda12.1, cuda12.4, npu |
| `--cuda VER` | CUDA version shorthand (11.8, 12.1, 12.4) |
| `--npu` | Enable Ascend NPU support |
| `--torch-npu VER` | torch-npu version (default: 2.5.1) |
| `--extras EXTRAS` | Optional deps: training, simulation, tensorflow, all |
| `--training` | Shorthand for --extras training |
| `--all` | Install all optional dependencies |

**Dependency Groups:**

| Group | Packages | Use Case |
|-------|----------|----------|
| (none) | Core only | Data collection, robot control (fastest) |
| `training` | diffusers, wandb, matplotlib, numba | Policy training |
| `simulation` | gymnasium, pymunk, gym-pusht | Simulation environments |
| `tensorflow` | tensorflow, tensorflow-datasets | TF dataset formats |
| `all` | Everything | Full installation |

### Manual Environment Setup (Alternative)

#### 1.1 Initialize DoRobot Environment

```bash
# Create and activate conda environment
conda create --name dorobot python==3.11
conda activate dorobot

# Install the project (choose one)
pip install -e .                    # Core only (fastest, for data collection)
pip install -e ".[training]"        # Core + training dependencies
pip install -e ".[simulation]"      # Core + simulation environments
pip install -e ".[all]"             # Everything

# Install DORA-RS
pip install dora-rs-cli

# Install robot dependencies
cd operating_platform/robot/robots/so101_v1 && pip install -e .
cd operating_platform/robot/components/arm_normal_so101_v1 && pip install -e .
```

#### 1.2 Install PyTorch (Choose Your Platform)

**CUDA:**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Ascend NPU (310B):**
```bash
# Install PyTorch 2.5.1 (CPU version, compatible with torch-npu)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install torch-npu
pip install torch-npu==2.5.1
```

> **NPU Prerequisites:** CANN toolkit must be installed. Visit [Huawei Ascend](https://www.hiascend.com/software/cann) for installation instructions.

#### 1.3 Install System Dependencies (Linux)

```bash
sudo apt install libportaudio2
```

## SO101 Robot Operations

### 2.1 Calibrate SO101 Arm

Calibration files are stored in `arm_normal_so101_v1/.calibration`

**Calibrate Arm 1:**
```bash
cd operating_platform/robot/components/arm_normal_so101_v1/

# Calibrate leader arm 1
dora run dora_calibrate_leader.yml

# Calibrate follower arm 1
dora run dora_calibrate_follower.yml
```

**Calibrate Arm 2:**
```bash
cd operating_platform/robot/components/arm_normal_so101_v1/

# Calibrate leader arm 2
dora run dora_calibrate_leader2.yml

# Calibrate follower arm 2
dora run dora_calibrate_follower2.yml
```

### 2.2 Teleoperate SO101 Arm

```bash
cd operating_platform/robot/components/arm_normal_so101_v1/
dora run dora_teleoperate_arm.yml
```

## Data Recording

### 3.1 Hardware Connection Order

**Important:** Follow this order to ensure correct device indices.

1. **Disconnect all devices** (cameras and robotic arms)

2. **Connect head camera first:**
   ```bash
   ls /dev/video*
   # Should see: /dev/video0 /dev/video1
   ```

3. **Connect wrist camera:**
   ```bash
   ls /dev/video*
   # Should see: /dev/video0 /dev/video1 /dev/video2 /dev/video3
   ```

4. **Connect leader arm:**
   ```bash
   ls /dev/ttyACM*
   # Should see: /dev/ttyACM0
   ```

5. **Connect follower arm:**
   ```bash
   ls /dev/ttyACM*
   # Should see: /dev/ttyACM0 /dev/ttyACM1
   ```

### 3.2 Start Data Collection

**Single Command (Recommended):**
```bash
# Basic usage - starts both DORA and CLI automatically
bash scripts/run_so101.sh

# With custom dataset name
REPO_ID=my-dataset bash scripts/run_so101.sh

# With custom task description
REPO_ID=my-dataset SINGLE_TASK="pick up the cube" bash scripts/run_so101.sh

# With Ascend NPU support
USE_NPU=1 bash scripts/run_so101.sh
```

**Manual Two-Terminal Method (Alternative):**

Terminal 1 - Start DORA dataflow:
```bash
conda activate dorobot
cd operating_platform/robot/robots/so101_v1
dora run dora_teleoperate_dataflow.yml
```

Terminal 2 - Start recording CLI:
```bash
conda activate dorobot
bash scripts/run_so101_cli.sh
```

### 3.3 Recording Controls

| Key | Action |
|-----|--------|
| `n` | Save current episode and start new one |
| `e` | Stop recording and exit |

## Training

```bash
conda activate dorobot

python operating_platform/core/train.py \
  --dataset.repo_id="/path/to/dataset" \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false
```

**For NPU training:**
```bash
python operating_platform/core/train.py \
  --dataset.repo_id="/path/to/dataset" \
  --policy.type=act \
  --policy.device=npu \
  ...
```

## Inference

```bash
conda activate dorobot

python operating_platform/core/inference.py \
  --robot.type=so101 \
  --inference.single_task="task description" \
  --inference.dataset.repo_id="/path/to/dataset" \
  --policy.path="/path/to/checkpoint/pretrained_model"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONDA_ENV` | `dorobot` | Conda environment name |
| `REPO_ID` | `so101-test` | Dataset repository ID |
| `SINGLE_TASK` | `start and test so101 arm.` | Task description |
| `USE_NPU` | `0` | Set to `1` for Ascend NPU support |
| `ASCEND_TOOLKIT_PATH` | `/usr/local/Ascend/ascend-toolkit` | CANN toolkit path |

## Project Structure

```
DoRobotTest/
├── operating_platform/
│   ├── core/           # Main pipelines (record, train, inference)
│   ├── robot/          # Robot hardware abstraction
│   │   ├── robots/     # Robot configurations (so101_v1, aloha_v1)
│   │   └── components/ # Hardware components (arms, cameras)
│   ├── policy/         # Policy implementations (ACT, Diffusion, etc.)
│   ├── dataset/        # Dataset management
│   └── utils/          # Utility functions
├── scripts/            # Launch scripts
│   ├── setup_env.sh    # Environment setup
│   ├── run_so101.sh    # Unified launcher
│   └── run_so101_cli.sh
└── docs/               # Documentation
```

## Acknowledgment

- LeRobot: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- DORA-RS: [https://github.com/dora-rs/dora](https://github.com/dora-rs/dora)
