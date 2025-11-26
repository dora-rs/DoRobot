import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

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


RERUN_WEB_PORT = 9195
RERUN_WS_PORT = 9285


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: Robot
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = False


@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    # logging.info(pformat(asdict(cfg)))

    # robot = make_robot_from_config(cfg.robot)
    dataset = DoRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.hf_dataset.select_columns("action")
    # robot.connect()
    robot = cfg.robot

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action_array = actions[idx]["action"]
        action = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            action[name] = action_array[i]

        # print(f"action: {action}")
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / dataset.fps - dt_s)

    # robot.disconnect()


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    replay: DatasetReplayConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]

@parser.wrap()
def main(cfg: ControlPipelineConfig):
    init_logging()
    git_branch_log()
    logging.info(pformat(asdict(cfg)))

    daemon = Daemon(fps=cfg.replay.fps)
    daemon.start(cfg.robot)

    replay_cfg = ReplayConfig(daemon.robot, cfg.replay)
    dataset = DoRobotDataset(cfg.replay.repo_id, root=cfg.replay.root)

    # 用于线程间通信的异常队列
    error_queue = queue.Queue()
    # 用于通知replay线程停止的事件
    stop_event = threading.Event()

    def visual_worker():
        """visual工作线程函数"""
        try:
            # 主线程执行可视化（阻塞直到窗口关闭或超时）
            visualize_dataset(
                dataset,
                mode="distant",
                episode_index=cfg.replay.episode,
                web_port=RERUN_WEB_PORT,
                ws_port=RERUN_WS_PORT,
                stop_event=stop_event  # 需要replay函数支持stop_event参数
            )
        except Exception as e:
            error_queue.put(e)

    # 创建并启动replay线程
    visual_thread = threading.Thread(
        target=visual_worker,
        name="VisualThread",
        daemon=True  # 设置为守护线程，主程序退出时自动终止
    )
    visual_thread.start()

    print(f"Visual at: http://localhost:{RERUN_WEB_PORT}/?url=ws://localhost:{RERUN_WS_PORT}")

    try:
        replay(replay_cfg)
    finally:
        # 无论可视化是否正常结束，都通知replay线程停止
        stop_event.set()
        # 等待replay线程安全退出（设置合理超时）
        visual_thread.join(timeout=5.0)
        
        # 检查线程是否已退出
        if visual_thread.is_alive():
            print("Warning: Visual thread did not exit cleanly")
        
        # 处理子线程异常
        try:
            error = error_queue.get_nowait()
            raise RuntimeError(f"Visual failed in thread: {str(error)}") from error
        except queue.Empty:
            pass
            
    print("="*20)
    print("Replay Complete Success!")
    print("="*20)
    
if __name__ == "__main__":
    main()
