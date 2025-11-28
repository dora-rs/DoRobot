# DoRobot-Test

>  Dora LeRobot  Version

## 0. Start (with Docker) coming soon

<!-- get this project

```sh
git cloen https://github.com/DoRobot-Project/Operating-Platform.git
cd Operating-Platform
```

build docker image
```sh
docker build -f docker/Dockerfile.base -t operating-platform:V1.0 .
```

make dir
```sh
mkdir /data/hf
```

run sh
```sh
sh docker/start.sh
```


[tool.uv.sources]
lerobot_lite = { path = "operating_platform/lerobot_lite"} -->

## 1. Start (without Docker)

get this project

```sh
git clone https://github.com/ustbxcj/DoRobotTest.git
cd DoRobotTest
```

### 1.1. Initital DoRobot enviroment

creat conda env

```sh
conda create --name op python==3.11
```

activate conda env

```sh
conda activate op
```

install this project

```sh
pip install -e .
```

**install pytorch, according to your platform**

```sh
# ROCM 6.1 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.1
# ROCM 6.2.4 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

install libportaudio2

```
sudo apt install libportaudio2
```

### 1.2. Initital SO101 enviroment

Open a new terminal and switch to the DoRobot-Preview project directory.

creat conda env

```sh
conda create --name dr-robot-so101 python==3.10
```

activate conda env

```sh
conda activate dr-robot-so101
```

install robot enviroment

```sh
cd operating_platform/robot/robots/so101_v1
pip install -e .
```

### 1.3. Calibrate SO101 Arm
对1号臂进行矫正
```
calibrate leader arm1---矫正1号臂主动臂
```
cd operating_platform/robot/components/arm_normal_so101_v1/
dora run dora_calibrate_leader.yml
```

calibrate follower arm1---矫正1号臂从动臂
```
cd operating_platform/robot/components/arm_normal_so101_v1/
dora run dora_calibrate_follower.yml
```


对2号臂进行矫正
```
calibrate leader arm2---矫正2号臂主动臂
```
cd operating_platform/robot/components/arm_normal_so101_v1/
dora run dora_calibrate_leader2.yml
```

calibrate follower arm2---矫正2号臂从动臂
```
cd operating_platform/robot/components/arm_normal_so101_v1/
dora run dora_calibrate_follower2.yml
```
矫正文件存储在arm_normal_so101_v1/.calibration
## 2. Teleoperate SO101 Arm

```
cd operating_platform/robot/components/arm_normal_so101_v1/
dora run dora_teleoperate_arm.yml
```

## 3. Record Data

You need to unplug all camera and robotic arm data interfaces first, then plug in the head camera.

```
ls /dev/video*
```

you can see:

```
/dev/video0 /dev/video1
```

If you see other indices, please make sure that all other cameras have been disconnected from the computer. If you are unable to remove them, please modify the camera index in the YAML file. 

then plug in the head camera.

```
ls /dev/video*
```

you can see:

```
/dev/video0 /dev/video1 /dev/video2 /dev/video3
```

now, you finish camera connect.

Next, connect the robotic arm by first plugging in the leader arm's USB interface.

```
ls /dev/ttyACM*
```

you can see:

```
/dev/ttyACM0
```

Then plugging in the follower arm's USB interface.

```
ls /dev/ttyACM*
```

you can see:

```
/dev/ttyACM0 /dev/ttyACM1
```

run dora data flow 

```
cd operating_platform/robot/robots/so101_v1
conda activate dr-robot-so101
dora run dora_teleoperate_dataflow.yml
```

Open a new terminal, then:

```
bash scripts/run_so101_cli.sh
```

You can modify the task name and task description by adjusting the parameters within the run_so101_cli.sh file.




# Acknowledgment
 - LeRobot 🤗: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
