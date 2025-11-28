import cv2
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from functools import cache
import traceback
import draccus

from operating_platform.core.daemon import Daemon

from operating_platform.dataset.dorobot_dataset import DoRobotDataset
from operating_platform.robot.robots.utils import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    busy_wait
)
from operating_platform.utils.utils import (
    init_logging,
    log_say,
    get_current_git_branch,
    git_branch_log
)
from operating_platform.utils import parser
import queue
import threading
from operating_platform.dataset.visual.visual_dataset import visualize_dataset
from operating_platform.policy.factory import make_policy
from operating_platform.utils.dataset import (
    build_dataset_frame,
    hw_to_dataset_features,
)
from operating_platform.policy.pretrained import PreTrainedPolicy
from contextlib import nullcontext
import torch
import numpy as np
from copy import copy
from operating_platform.utils.utils import get_safe_torch_device
from operating_platform.config.policies import PreTrainedConfig


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True

@dataclass
class DatasetInferenceConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None


@dataclass
class InferenceConfig:
    dataset: DatasetInferenceConfig
    fps: int = 30
    policy_path: str | Path | None = None
    countdown_seconds: int = 3
    single_task: str = "TEST: no task description. Example: Pick apple."


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

# @draccus.wrap()
def inference(cfg: InferenceConfig, policy_cfg: PreTrainedConfig,daemon: Daemon):
    dataset = DoRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
    )

    # if hasattr(robot, "cameras") and len(robot.cameras) > 0:
    #     dataset.start_image_writer(
    #         num_processes=cfg.dataset.num_image_writer_processes,
    #         num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
    #     )
    # sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)

    policy = None if policy_cfg is None else make_policy(policy_cfg, ds_meta=dataset.meta)
    if policy is None:
        logging.error("Policy cannot be None")

    while True:
        logging.info("="*30)
        logging.info(f"Starting inference")
        logging.info("="*30)

        if policy is not None:
            policy.reset()
        
        # 8. 开始记录（带倒计时）
        if cfg.countdown_seconds > 0:
            for i in range(cfg.countdown_seconds, 0, -1):
                logging.info(f"Recording starts in {i}...")
                time.sleep(1)
        
        # record.start()
        
        # 9. 用户交互循环（改进的输入处理）
        logging.info("Recording active. Press:")
        logging.info("- 'n' to finish current episode and start new one")
        logging.info("- 'e' to stop recording and exit")
        
        while True:
            daemon.update()
            observation = daemon.get_observation()

            if policy is not None or dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            if policy is not None:
                action_values = predict_action(
                    observation_frame,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    task=cfg.single_task,
                    robot_type=daemon.robot.robot_type,
                )
                action = {key: action_values[i].item() for i, key in enumerate(daemon.robot.action_features)}
                print(f"action:{action}")
                daemon.robot.send_action(action)
            
            # 显示图像（仅在非无头模式）
            if observation and not is_headless():
                for key in observation:
                    if "image" in key:
                        img = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Camera: {key}", img)
            
            # 处理用户输入
            key = cv2.waitKey(1)  # 增加延迟减少CPU占用
            if key in [ord('n'), ord('N')]:
                logging.info("Ending current episode...")
                break
            elif key in [ord('e'), ord('E')]:
                logging.info("Stopping recording and exiting...")
                # record.stop()
                # record.save()
                return  # 直接退出函数
        
        # 10. 保存当前episode
        # record.stop()
        # record.save()
        # logging.info(f"Episode saved. Total episodes: {record.dataset.meta.total_episodes}")
        
        # 11. 环境重置（带超时和可视化）
        logging.info("*"*30)
        logging.info("Resetting environment - Press 'p' to proceed")
        logging.info("Note: Robot will automatically reset in 10 seconds if no input")
        
        reset_start = time.time()
        reset_timeout = 60  # 10秒超时
        
        while time.time() - reset_start < reset_timeout:
            daemon.update()
            if observation := daemon.get_observation():
                for key in observation:
                    if "image" in key:
                        img = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Reset View: {key}", img)
            
            key = cv2.waitKey(10)
            if key in [ord('p'), ord('P')]:
                logging.info("Reset confirmed by user")
                break
            elif key in [ord('e'), ord('E')]:
                logging.info("User aborted during reset")
                return
        
        # 12. 清理窗口（仅在无新窗口时）
        if not is_headless():
            cv2.destroyAllWindows()
            logging.debug("Closed all OpenCV windows")


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    inference: InferenceConfig
    policy: PreTrainedConfig | None = None
    teleop: None = None 

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")
        
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

@parser.wrap()
def main(cfg: ControlPipelineConfig):
    init_logging(level=logging.INFO, force=True)
    git_branch_log()

    logging.info(pformat(asdict(cfg)))

    daemon = Daemon(fps=cfg.inference.fps)
    daemon.start(cfg.robot)
    daemon.update()

    try:
        inference(cfg.inference, cfg.policy, daemon)
            
    except KeyboardInterrupt:
        print("coordinator and daemon stop")

    finally:
        daemon.stop()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
