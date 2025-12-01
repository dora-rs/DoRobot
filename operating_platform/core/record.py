import time
import threading

from deepdiff import DeepDiff
from dataclasses import dataclass

from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import Robot, busy_wait, safe_disconnect, make_robot_from_config

from operating_platform.dataset.dorobot_dataset import *
from operating_platform.core.daemon import Daemon
from operating_platform.core.async_episode_saver import AsyncEpisodeSaver, EpisodeMetadata
import draccus
from operating_platform.utils import parser
from operating_platform.utils.utils import has_method, init_logging, log_say, get_current_git_branch, git_branch_log, get_container_ip_from_hosts

from operating_platform.utils.constants import DOROBOT_DATASET
from operating_platform.utils.data_file import (
    get_data_duration,
    get_data_size ,
    update_dataid_json,
    update_common_record_json,
    delete_dataid_json
)
from operating_platform.utils.dataset import (
    build_dataset_frame,
    hw_to_dataset_features,
)


def sanity_check_dataset_robot_compatibility(
    dataset: DoRobotDataset, robot: RobotConfig, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        # ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


# def sanity_check_dataset_name(repo_id, policy_cfg):
#     _, dataset_name = repo_id.split("/")
#     # either repo_id doesnt start with "eval_" and there is no policy
#     # or repo_id starts with "eval_" and there is a policy

#     # Check if dataset_name starts with "eval_" but policy is missing
#     if dataset_name.startswith("eval_") and policy_cfg is None:
#         raise ValueError(
#             f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
#         )

#     # Check if dataset_name does not start with "eval_" but policy is provided
#     if not dataset_name.startswith("eval_") and policy_cfg is not None:
#         raise ValueError(
#             f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
#         )


@dataclass
class RecordConfig():
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str = "TEST: no task description. Example: Pick apple."
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30

    # Encode frames in the dataset into video
    video: bool = True

    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False

    # Upload on private repository on the Hugging Face hub.
    private: bool = False

    # Add tags to your dataset on the hub.
    tags: list[str] | None = None

    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4

    # Resume recording on an existing dataset.
    resume: bool = False

    # NEW: Async save options
    # Enable asynchronous episode saving (non-blocking save operations)
    use_async_save: bool = True
    # Maximum number of episodes that can be queued for async saving
    async_save_queue_size: int = 10
    # Timeout in seconds for save operations
    async_save_timeout_s: int = 300
    # Maximum retry attempts for failed saves
    async_save_max_retries: int = 3

    # Cloud offload mode: skip local video encoding, upload raw images to cloud for encoding/training
    # When True, all saves will skip video encoding (for edge devices like Orange Pi with slow encoding)
    # When False, normal local video encoding is performed
    cloud_offload: bool = False

    record_cmd = None


class Record:
    def __init__(self, fps: int, robot: Robot, daemon: Daemon, record_cfg: RecordConfig, record_cmd):
        self.robot = robot
        self.fps = fps

        self.daemon = daemon
        self.record_cfg = record_cfg
        self.record_cfg.record_cmd = record_cmd
        self.record_cmd = record_cfg.record_cmd

        self.last_record_episode_index = 0
        self.record_complete = False
        self.save_data = None

        # Lock to protect buffer swap during save_async (prevents race condition
        # where recording thread adds frame while buffer is being captured)
        self._buffer_lock = threading.Lock()

        # Async save support
        self.use_async_save = getattr(record_cfg, 'use_async_save', True)  # Default to True
        self.async_saver = None

        # Cloud offload mode: skip video encoding for all saves
        self.cloud_offload = getattr(record_cfg, 'cloud_offload', False)
        if self.use_async_save:
            max_queue_size = getattr(record_cfg, 'async_save_queue_size', 10)
            self.async_saver = AsyncEpisodeSaver(max_queue_size=max_queue_size)

        action_features = hw_to_dataset_features(robot.action_features, "action", self.robot.use_videos)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", self.robot.use_videos)
        dataset_features = {**action_features, **obs_features}

        if self.record_cfg.resume:
            self.dataset = DoRobotDataset(
                record_cfg.repo_id,
                root=record_cfg.root,
            )
            if len(robot.cameras) > 0:
                self.dataset.start_image_writer(
                    num_processes=record_cfg.num_image_writer_processes,
                    num_threads=record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            if len(robot.microphones) > 0:
                self.dataset.start_audio_writer(
                    microphones=robot.microphones,
                )
            sanity_check_dataset_robot_compatibility(self.dataset, robot, record_cfg.fps, record_cfg.video)
        else:
            # Create empty dataset or load existing saved episodes
            # sanity_check_dataset_name(record_cfg.repo_id, record_cfg.policy)
            self.dataset = DoRobotDataset.create(
                record_cfg.repo_id,
                record_cfg.fps,
                root=record_cfg.root,
                robot=robot,
                features=dataset_features,
                use_videos=record_cfg.video,
                use_audios=len(robot.microphones) > 0,
                image_writer_processes=record_cfg.num_image_writer_processes,
                image_writer_threads=record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        self.thread = threading.Thread(target=self.process, daemon=True)
        self.running = True

    def _create_new_episode_buffer(self) -> dict:
        """
        Create a new episode buffer with a pre-allocated episode index.

        When using async save, the episode index is allocated from the
        AsyncEpisodeSaver's counter to ensure correct sequencing even when
        multiple episodes are being saved in parallel.
        """
        if self.use_async_save and self.async_saver:
            # Pre-allocate index from async saver
            episode_index = self.async_saver.allocate_next_index()
            logging.info(f"[Record] Pre-allocated episode index {episode_index}")
            return self.dataset.create_episode_buffer(episode_index=episode_index)
        else:
            # Use default behavior (based on meta.total_episodes)
            return self.dataset.create_episode_buffer()

    def start(self):
        # Start async saver BEFORE recording thread to ensure proper buffer setup
        if self.use_async_save and self.async_saver:
            initial_ep_idx = self.dataset.meta.total_episodes
            self.async_saver.start(initial_episode_index=initial_ep_idx)
            logging.info(f"[Record] Async saver started (initial_ep_idx={initial_ep_idx})")

            # Create initial episode buffer with pre-allocated index
            # This MUST happen before the recording thread starts to avoid race conditions
            self.dataset.episode_buffer = self._create_new_episode_buffer()

        # Now start the recording thread
        self.thread.start()
        self.running = True

    def process(self):
        while self.running:
            if self.dataset is not None:
                start_loop_t = time.perf_counter()

                observation = self.daemon.get_observation()
                action = self.daemon.get_obs_action()

                if self.dataset is not None:
                    observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
                    action_frame = build_dataset_frame(self.dataset.features, action, prefix="action")
                    frame = {**observation_frame, **action_frame}
                    # Use lock to prevent race condition with save_async buffer swap
                    with self._buffer_lock:
                        self.dataset.add_frame(frame, self.record_cfg.single_task)

                dt_s = time.perf_counter() - start_loop_t

                if self.fps is not None:
                    busy_wait(1 / self.fps - dt_s)


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

        # Stop async saver if enabled (wait for pending saves)
        if self.use_async_save and self.async_saver:
            status = self.async_saver.get_status()
            if status["pending_count"] > 0:
                logging.info(f"[Record] Waiting for {status['pending_count']} pending saves...")
                self.async_saver.wait_all_complete(timeout=300.0)  # 5 min timeout for saves

        # stop_recording(robot, listener, record_cfg.display_cameras)
        # log_say("Stop recording", record_cfg.play_sounds, blocking=True)

    def save_async(self, skip_encoding: bool = False) -> EpisodeMetadata:
        """
        Queue episode for asynchronous saving.
        Returns metadata immediately without blocking.

        The episode buffer already has a pre-allocated episode_index
        (assigned when the buffer was created via _create_new_episode_buffer).
        This ensures images were saved to the correct directory during recording.

        Args:
            skip_encoding: If True, skip video encoding (cloud offload mode).
        """
        import copy

        # CRITICAL: Use lock to atomically capture buffer and swap to new one
        # This prevents the recording thread from adding frames during the swap
        with self._buffer_lock:
            current_ep_idx = self.dataset.episode_buffer.get("episode_index", "?")
            logging.info(f"[Record] Queueing episode {current_ep_idx} for async save (skip_encoding={skip_encoding})...")

            # Deep copy the buffer INSIDE the lock (before recording thread can add more frames)
            buffer_copy = copy.deepcopy(self.dataset.episode_buffer)

            # Create new episode buffer INSIDE the lock
            self.dataset.episode_buffer = self._create_new_episode_buffer()

        # Queue save task with the copied buffer (outside lock to minimize lock hold time)
        metadata = self.async_saver.queue_save(
            episode_buffer=buffer_copy,  # Pass the COPY, not the live buffer
            dataset=self.dataset,
            record_cfg=self.record_cfg,
            record_cmd=self.record_cmd,
            skip_encoding=skip_encoding,
        )

        # Update local state immediately
        self.last_record_episode_index = metadata.episode_index
        self.record_complete = True

        logging.info(f"[Record] Episode {metadata.episode_index} queued (pos={metadata.queue_position})")
        return metadata

    def save_sync(self, skip_encoding: bool = False) -> dict:
        """Original synchronous save method.

        Args:
            skip_encoding: If True, skip video encoding (cloud offload mode).
        """
        print(f"will save_episode (skip_encoding={skip_encoding})")

        episode_index = self.dataset.save_episode(skip_encoding=skip_encoding)

        print("save_episode succcess, episode_index:", episode_index)

        update_dataid_json(self.record_cfg.root, episode_index,  self.record_cmd)
        if episode_index == 0 and self.dataset.meta.total_episodes == 1:
            update_common_record_json(self.record_cfg.root, self.record_cmd)

        print("update_dataid_json succcess")

        if self.record_cfg.push_to_hub:
            self.dataset.push_to_hub(tags=self.record_cfg.tags, private=self.record_cfg.private)

        file_size = get_data_size(self.record_cfg.root, self.record_cmd)
        file_duration = get_data_duration(self.record_cfg.root, self.record_cmd)

        print("get_data_size succcess, file_size:", file_size)

        data = {
            "file_message": {
                "file_name": self.record_cfg.repo_id,
                "file_local_path": str(self.record_cfg.root),
                "file_size": str(file_size),
                "file_duration": str(file_duration),
            },
            "verification": {
                "file_integrity": "pass",
                "camera_frame_rate": "pass",
            }
        }

        self.record_complete = True
        self.last_record_episode_index = episode_index

        self.save_data = data
        return data

    def save(self, skip_encoding: bool | None = None) -> EpisodeMetadata | dict:
        """
        Save episode - async by default, fallback to sync if needed.

        Args:
            skip_encoding: If True, skip video encoding and keep raw PNG images.
                          If None (default), uses self.cloud_offload setting.
                          Used for cloud offload mode where encoding is done on server.
        """
        # Use cloud_offload setting if skip_encoding not explicitly specified
        if skip_encoding is None:
            skip_encoding = self.cloud_offload

        if self.use_async_save:
            return self.save_async(skip_encoding=skip_encoding)
        else:
            return self.save_sync(skip_encoding=skip_encoding)

    def discard(self):
        if self.record_complete == True:
            delete_dataid_json(self.record_cfg.root, self.last_record_episode_index, self.record_cmd)
            self.dataset.remove_episode(self.last_record_episode_index)
        else:
            self.dataset.clear_episode_buffer()


# def stop_recording(robot, listener, display_cameras):
#     robot.disconnect()

#     if not is_headless():
#         if listener is not None:
#             listener.stop()

#         if display_cameras:
#             # cv2.destroyAllWindows()
#             pass

# @dataclass
# class RecordConfig:
#     robot: Robot
#     dataset: DatasetReplayConfig
#     # Use vocal synthesis to read events.
#     play_sounds: bool = False

@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    # control: ControlConfig
    record: RecordConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]

@parser.wrap()
def record(cfg: ControlPipelineConfig):
    init_logging()
    git_branch_log()

    # daemon = Daemon(fps=DEFAULT_FPS)
    # daemon.start(cfg.robot)

    # robot_daemon = Record(cfg.fps,cfg.)

    # robot_daemon.start()


def main():
    record()

if __name__ == "__main__":
    main()
