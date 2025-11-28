
import cv2
import json
import time
import draccus
import socketio
import requests
import traceback
import threading
import queue
import subprocess
import os
from pathlib import Path

from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pformat
from deepdiff import DeepDiff
from functools import cache
from termcolor import colored
from datetime import datetime


# from operating_platform.policy.config import PreTrainedConfig
from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import make_robot_from_config, Robot, busy_wait, safe_disconnect
from operating_platform.utils import parser
from operating_platform.utils.utils import has_method, init_logging, log_say, get_current_git_branch, git_branch_log, get_container_ip_from_hosts
from operating_platform.utils.data_file import find_epindex_from_dataid_json

from operating_platform.utils.constants import DOROBOT_DATASET
from operating_platform.dataset.dorobot_dataset import *
from operating_platform.dataset.visual.visual_dataset import visualize_dataset

# from operating_platform.core._client import Coordinator
from operating_platform.core.daemon import Daemon
from operating_platform.core.record import Record, RecordConfig
from operating_platform.core.replay import DatasetReplayConfig, ReplayConfig, replay
from operating_platform.utils.camera_display import CameraDisplay

DEFAULT_FPS = 30

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
class ControlPipelineConfig:
    robot: RobotConfig
    record: RecordConfig
    # control: ControlConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]
#自己写了份初稿发现可以运行，采用AI润色完善代码
class VideoEncoderThread(threading.Thread):
    """
    后台视频编码守护线程：
    - 自动从任务队列读取任务
    - 每个任务使用 ffmpeg 将图片序列编码为 mp4 视频
    - 支持多线程并发加速编码
    """

    def __init__(self, num_workers: int = 3):
        """
        :param num_workers: 并发 ffmpeg 编码线程数（建议 2~4）
        """
        super().__init__(daemon=True)
        self.task_queue = queue.Queue()
        self.running = True
        self.num_workers = num_workers
        self.workers: list[threading.Thread] = []

    def run(self):
        """主线程启动所有 worker 并维持运行"""
        print(f"[VideoEncoderThread] Starting with {self.num_workers} workers...")
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"EncoderWorker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

        # 主线程只是负责维持生命周期
        while self.running:
            time.sleep(0.5)

    def _worker_loop(self):
        """每个 worker 从队列中拉取任务并执行"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                if task is not None:
                    self.encode_video(**task)
            except Exception as e:
                print(f"[{threading.current_thread().name}] Error: {e}")
            finally:
                self.task_queue.task_done()

    def encode_video(self, img_dir: Path, output_path: Path, fps: int = 30):
        """
        使用 ffmpeg 将指定文件夹下的图片编码为视频
        """
        if not img_dir.exists():
            print(f"[VideoEncoderThread] Directory not found: {img_dir}")
            return

        images = sorted([p for p in img_dir.glob("*.png")])
        if not images:
            print(f"[VideoEncoderThread] No images found in {img_dir}")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{threading.current_thread().name}] Encoding {len(images)} frames -> {output_path}")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", "*.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                cwd=str(img_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            print(f"[{threading.current_thread().name}] Finished: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[{threading.current_thread().name}] ffmpeg failed for {img_dir}: {e}")

    def add_task(self, img_dir: Path, output_path: Path, fps: int = 30):
        """添加编码任务"""
        self.task_queue.put({"img_dir": img_dir, "output_path": output_path, "fps": fps})

    def stop(self):
        """停止所有线程（不等待队列）"""
        print("[VideoEncoderThread] Stopping encoder threads...")
        self.running = False
        # 给每个worker一个None任务，确保其能退出阻塞
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        print("[VideoEncoderThread] Stop signal sent to workers.")
    def is_idle(self) -> bool:
        """
        检查编码器是否空闲：
        - 队列为空且所有 ffmpeg 子进程执行完毕
        """
        return self.task_queue.empty()
    
# def record_loop(cfg: ControlPipelineConfig, daemon: Daemon, video_encoder:VideoEncoderThread):
def record_loop(cfg: ControlPipelineConfig, daemon: Daemon):

    # 确保数据集根目录存在
    dataset_path = DOROBOT_DATASET
    dataset_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Dataset root directory: {dataset_path}")

    # 1. 动态获取当前日期（支持跨天运行）
    date_str = datetime.now().strftime("%Y%m%d")
    repo_id = cfg.record.repo_id

    target_dir = dataset_path / date_str / "experimental" / repo_id
    # 创建目标目录（确保父目录存在）
    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Target directory: {target_dir}")

    # 检查恢复模式（更健壮的路径检查）
    resume = False
    if any(target_dir.iterdir()):  # 检查目录是否非空
        resume = True
        logging.info(f"Resuming recording in existing directory: {target_dir}")
    else:
        logging.info(f"Starting new recording session in: {target_dir}")

    # 任务配置（从配置获取而非硬编码）
    try:
        record_cmd = {
            "task_id": cfg.record.task_id or "default_task",
            "task_name": repo_id,
            "task_data_id": cfg.record.data_id or "001",
            "collector_id": cfg.record.collector_id or "default_collector",
            "countdown_seconds": cfg.record.countdown or 3,
            "task_steps": [
                {
                    "duration": str(step.get("duration", 10)),
                    "instruction": step.get("instruction", "put")
                } for step in cfg.record.task_steps
            ]
        }
    except Exception as e:
        logging.error(f"Invalid task configuration: {str(e)}")
        record_cmd = {
            "task_id": "fallback_task",
            "task_name": repo_id,
            "task_data_id": "001",
            "collector_id": "fallback_collector",
            "countdown_seconds": 3,
            "task_steps": [{"duration": "10", "instruction": "put"}]
        }
        logging.warning("Using fallback task configuration")

    # 创建记录器一次，在整个session中复用
    record_cfg = RecordConfig(
        fps=cfg.record.fps,
        repo_id=repo_id,
        single_task=cfg.record.single_task,
        video=daemon.robot.use_videos,
        resume=resume,
        root=target_dir,
        use_async_save=cfg.record.use_async_save,
        async_save_queue_size=cfg.record.async_save_queue_size,
        async_save_timeout_s=cfg.record.async_save_timeout_s,
        async_save_max_retries=cfg.record.async_save_max_retries,
    )
    record = Record(
        fps=cfg.record.fps,
        robot=daemon.robot,
        daemon=daemon,
        record_cfg=record_cfg,
        record_cmd=record_cmd
    )

    logging.info("="*30)
    logging.info(f"Starting recording session | Resume: {resume} | Episodes: {record.dataset.meta.total_episodes}")
    logging.info("="*30)

    # Create unified camera display (combines all cameras into one window)
    camera_display = CameraDisplay(
        window_name="Recording - Cameras",
        layout="horizontal",
        show_labels=True
    )

    # 开始记录（带倒计时）- 只做一次
    if record_cmd.get("countdown_seconds", 3) > 0:
        for i in range(record_cmd["countdown_seconds"], 0, -1):
            logging.info(f"Recording starts in {i}...")
            time.sleep(1)

    record.start()

    # Voice prompt: ready to start
    log_say("Ready to start. Press N to save and start next episode.", play_sounds=True)

    # 主循环：连续录制多个episodes
    while True:
        logging.info("Recording active. Press:")
        logging.info("- 'n' to save current episode and start new one")
        logging.info("- 'e' to stop recording and exit")

        # Episode录制循环
        while True:
            daemon.update()
            observation = daemon.get_observation()

            # 显示图像（仅在非无头模式）- 使用统一相机窗口
            if observation and not is_headless():
                key = camera_display.show(observation)
            else:
                key = cv2.waitKey(10)

            # 处理用户输入
            if key in [ord('n'), ord('N')]:
                logging.info("Saving current episode and starting new one...")

                # Save current episode (non-blocking)
                metadata = record.save()

                # Log save status
                if hasattr(metadata, 'episode_index'):
                    logging.info(f"Episode {metadata.episode_index} queued (queue pos: {metadata.queue_position})")

                # Check for save errors from previous episodes
                if hasattr(record, 'async_saver') and record.async_saver:
                    status = record.async_saver.get_status()
                    if status["failed_count"] > 0:
                        logging.warning(f"Warning: {status['failed_count']} episodes failed to save: "
                                       f"{status['failed_episodes']}")

                logging.info("*"*30)
                logging.info("Starting new episode...")
                logging.info("*"*30)

                # Voice prompt: recording new episode
                next_episode = record.dataset.meta.total_episodes
                log_say(f"Recording episode {next_episode}.", play_sounds=True)

                break  # Break to restart episode loop

            elif key in [ord('e'), ord('E')]:
                logging.info("Stopping recording and exiting...")

                # Voice prompt: end collection
                log_say("End collection. Please wait for video encoding.", play_sounds=True)

                # IMPORTANT: Stop the DORA daemon FIRST to disconnect hardware gracefully
                # This prevents hardware disconnection errors during save operations
                logging.info("Stopping DORA daemon (disconnecting hardware)...")
                daemon.stop()
                logging.info("DORA daemon stopped")

                # Save the current episode (async save queues it)
                metadata = record.save()
                if hasattr(metadata, 'episode_index'):
                    logging.info(f"Episode {metadata.episode_index} queued for saving")

                # Now stop recording thread and wait for image writer to complete
                # Hardware is already disconnected, so no risk of arm errors
                record.stop()

                # Properly shutdown async saver (waits for queue AND pending, then stops worker)
                if hasattr(record, 'async_saver') and record.async_saver:
                    status = record.async_saver.get_status()
                    pending_total = status['pending_count'] + status['queue_size']
                    if pending_total > 0:
                        logging.info(f"Waiting for {pending_total} saves (queue={status['queue_size']}, pending={status['pending_count']})...")

                    # stop() calls wait_all_complete() internally and properly shuts down worker thread
                    record.async_saver.stop(wait_for_completion=True)

                    # Print final status
                    final_status = record.async_saver.get_status()
                    logging.info(f"Save stats: queued={final_status['stats']['total_queued']} "
                               f"completed={final_status['stats']['total_completed']} "
                               f"failed={final_status['stats']['total_failed']}")

                return

        # Continue recording next episode (thread is still running, just save() reset the buffer)


@parser.wrap()
def main(cfg: ControlPipelineConfig):

    init_logging(level=logging.INFO, force=True)
    git_branch_log()
    logging.info(pformat(asdict(cfg)))

    daemon = Daemon(fps=DEFAULT_FPS)
    daemon.start(cfg.robot)
    daemon.update()

    # video_encoder = VideoEncoderThread()
    # video_encoder.start()

    try:
        # record_loop(cfg, daemon,video_encoder)
        record_loop(cfg, daemon)     
    except KeyboardInterrupt:
        print("coordinator and daemon stop")

    finally:
        daemon.stop()
        # video_encoder.stop()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
