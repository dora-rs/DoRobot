# Getting Started with Piper

## Installation (To do once)

```
git clone https://github.com/DoRobot-Project/Operating-Platform.git
cd test/piper
```

create conda env
```
conda create -n dr-piper-rerun python=3.11 -y
conda activate dr-piper-rerun
```

install dependencies
```
pip install .
pip install piper_sdk
```

```
dora build arms_only.yml
```

```
pip uninstall rerun-sdk -y
pip install rerun-sdk==0.23.4
```

## Use

```
dora run arms_only.yml
```
