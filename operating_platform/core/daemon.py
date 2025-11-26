# !!! Notice: daemon will combine with robot。TODO

import time
import torch
import logging
import threading

from typing import Any
from termcolor import colored

from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import make_robot_from_config, Robot, busy_wait, safe_disconnect


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        # for name in robot.leader_arms:
        #     key = f"read_leader_{name}_pos_dt_s"
        #     if key in robot.logs:
        #         log_dt(f"dt_R_leader_{name}", robot.logs[key])

        for name in robot.follower_arms:
        #     key = f"write_follower_{name}_goal_pos_dt_s"
        #     if key in robot.logs:
        #         log_dt(f"dt_W_foll_{name}", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt(f"dt_R_foll_{name}", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dt_R_camera_{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)



class Daemon:
    def __init__(self, fps: int | None = None):
        # self.record = False
        # self.evaluate
        self.fps = fps
        # self.display = display

        # self.thread = threading.Thread(target=self.process, daemon=True)
        self.running = True

        self.data_lock = threading.Lock()
        self.pre_action: Any | dict[str, torch.Tensor] = None
        self.obs_action: Any | dict[str, torch.Tensor] = None
        self.observation: Any | dict[str, torch.Tensor] = None


    def start(self, config: RobotConfig):
        try:
            self.robot = make_robot_from_config(config)
        except Exception as e:
            KeyboardInterrupt
        print("Make robot succese")

        if not self.robot.is_connected:
            self.robot.connect()
        print("Connect robot succese")

        # self.thread.start()
        # self.running = True

    
    def stop(self):
        # self.running = False
        # self.thread.join()
        pass

    def update(self):
        # while self.running:
        start_loop_t = time.perf_counter()

        observation, action = self.robot.teleop_step(record_data=True)

        # if observation is not None:
        #     self.observation = observation.copy()

        # if action is not None:
        #     self.obs_action = action.copy()
        self.set_observation(observation)
        self.set_obs_action(action)


        pre_action = self.get_pre_action()
        if pre_action is not None:
            # pre_action = self.pre_action.copy()
            action = self.robot.send_action(pre_action["action"])
            action = {"action": action}
        
        dt_s = time.perf_counter() - start_loop_t

        if self.fps is not None:
            busy_wait(1 / self.fps - dt_s)

        log_control_info(self.robot, dt_s, fps=self.fps)


    @property
    def cameras_info(self):
        cameras = {}
        for name, camera in self.robot.cameras.items():
            cameras[name] = camera.camera_index
        return cameras
    
    
    def set_pre_action(self, value: Any | dict[str, torch.Tensor]):
        with self.data_lock:
            if value is None:
                return
            self.pre_action = value.copy()
    
    def set_obs_action(self, value: Any | dict[str, torch.Tensor]):
        with self.data_lock:
            if value is None:
                return
            self.obs_action = value.copy()

    def set_observation(self, value: Any | dict[str, torch.Tensor]):
        with self.data_lock:
            if value is None:
                return
            self.observation = value.copy()

    def get_pre_action(self) -> Any | dict[str, torch.Tensor]:
        with self.data_lock:
            if self.pre_action is None:
                return None
            return self.pre_action.copy()

    def get_obs_action(self) -> Any | dict[str, torch.Tensor]:
        with self.data_lock:
            if self.obs_action is None:
                return None
            return self.obs_action.copy()
    
    def get_observation(self) -> Any | dict[str, torch.Tensor]:
        with self.data_lock:
            if self.observation is None:
                return None
            return self.observation.copy()


def daemon():
    robot_daemon = Daemon()

    robot_daemon.start()


def main():
    daemon()

if __name__ == "__main__":
    main()
