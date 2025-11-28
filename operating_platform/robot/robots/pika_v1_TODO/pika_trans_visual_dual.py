import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import cv2

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from operating_platform.robot.robots.pika_v1.pose_utils import xyzQuaternion2matrix, xyzrpy2Mat, matrixToXYZQuaternion


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
PIKA_RIGHT_NAME = os.getenv("PIKA_RIGHT_NAME", "right")
PIKA_LEFT_NAME = os.getenv("PIKA_LEFT_NAME", "left")
RETURN_IMAGE = os.getenv('RETURN_IMAGE', '1').lower() in ('1', 'true', 'yes')

def quat_slerp(q0, q1, t):
    """手动实现四元数 SLERP 插值"""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return (1 - t) * q0 + t * q1
    theta = np.arccos(dot) * t
    sin_theta = np.sin(theta)
    sin_full = np.sin(theta / t)
    return (np.sin(theta - theta / t) * q0 + sin_theta * q1) / sin_full

class TransformVisualizer:
    def __init__(self, axis_length=0.4, space_size=2):
        # 检查环境变量决定运行模式
        self.return_image = RETURN_IMAGE
        
        # 根据模式选择合适的 matplotlib 后端
        if self.return_image:
            # 无头模式：使用非交互式后端
            matplotlib.use('Agg')
        
        # 创建图形（无头模式下不会显示窗口）
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(azim=-160)
        self.axis_length = axis_length
        self.space_size = space_size
        
        # 存储多个设备的变换
        self.transforms = {}  # {device_id: transform_matrix}
        self.trajectory_points = {}  # {device_id: [points]}
        self.T_zero = None
        self.device_colors = {'left': 'cyan', 'right': 'magenta'}  # 设备显示颜色
        
        # 初始化绘图
        self._setup_plot()
        
        # 仅在交互模式下显示窗口
        if not self.return_image:
            plt.ion()
            plt.show()

    def _setup_plot(self):
        """初始化3D坐标系和图形设置"""
        base_T = np.eye(4)
        self.plot_transform(base_T, axis_length=2.0, label="Base Frame", color='black')
        
        # 设置坐标轴属性
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Dual Device 6D Visualization")
        self.ax.set_box_aspect([1, 1, 1])
        
        # 设置坐标范围
        limits = [-self.space_size, self.space_size]
        self.ax.set_xlim(limits)
        self.ax.set_ylim(limits)
        self.ax.set_zlim(limits)
        
        # 添加图例说明
        self.ax.legend([
            plt.Line2D([0], [0], color='cyan', lw=2),
            plt.Line2D([0], [0], color='magenta', lw=2),
            plt.Line2D([0], [0], color='black', lw=2)
        ], ['Left Device', 'Right Device', 'Base Frame'])
        
        plt.tight_layout()

    def plot_transform(self, T, axis_length=None, label=None, color='red'):
        """绘制单个坐标系变换，增加颜色参数"""
        axis_length = axis_length or self.axis_length
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * axis_length
        y_axis = origin + T[:3, 1] * axis_length
        z_axis = origin + T[:3, 2] * axis_length
        
        # 绘制轴线（使用传入的颜色）
        self.ax.plot([origin[0], x_axis[0]], 
                    [origin[1], x_axis[1]], 
                    [origin[2], x_axis[2]], 
                    c=color, linewidth=2, alpha=0.8)
        self.ax.plot([origin[0], y_axis[0]], 
                    [origin[1], y_axis[1]], 
                    [origin[2], y_axis[2]], 
                    c=color, linewidth=2, alpha=0.8)
        self.ax.plot([origin[0], z_axis[0]], 
                    [origin[1], z_axis[1]], 
                    [origin[2], z_axis[2]], 
                    c=color, linewidth=2, alpha=0.8)
        
        # 添加标签
        if label:
            self.ax.text(*origin, label, color='k', fontsize=8)

    def plot_trajectory(self, device_id, color='gray'):
        """绘制特定设备的轨迹"""
        if device_id not in self.trajectory_points or not self.trajectory_points[device_id]:
            return
            
        points = np.array(self.trajectory_points[device_id])
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                    color=color, linewidth=1, alpha=0.5)

    def update_visualization(self):
        """更新可视化界面，显示所有设备"""
        self.ax.cla()
        self._setup_plot()
        
        # 绘制所有设备的当前姿态
        for device_id, transform in self.transforms.items():
            if transform is not None:
                color = self.device_colors.get(device_id, 'red')
                label = f"{device_id.capitalize()} Pose"
                self.plot_transform(transform, label=label, color=color)
                # self.plot_trajectory(device_id, color=color)
        
        # 仅在交互模式下刷新
        if not self.return_image:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def set_initial_transform(self, T_initial):
        """设置初始变换作为基准"""
        self.T_zero = T_initial.copy()

    def update_transform(self, device_id, T_world):
        """更新指定设备的变换"""
        if self.T_zero is None:
            logger.warning("T_zero not set. Cannot update transform.")
            return
            
        # 计算相对变换
        T_relative = np.linalg.inv(self.T_zero) @ T_world
        
        # 更新变换数据
        self.transforms[device_id] = T_relative
        
        # 记录轨迹点
        if device_id not in self.trajectory_points:
            self.trajectory_points[device_id] = []
        self.trajectory_points[device_id].append(T_relative[:3, 3])
        
        # 更新可视化
        self.update_visualization()

        position = T_relative[:3, 3]
        rotation_matrix = T_relative[:3, :3]
        rotation = R.from_matrix(rotation_matrix).as_quat()
        
        # ===== 核心修改：根据模式返回不同内容 =====
        if self.return_image:
            # 确保图形已渲染
            self.fig.canvas.draw()
            
            # 从画布获取RGB图像数据
            # img_data = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            # img_data = img_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            width, height = self.fig.canvas.get_width_height()
            img_data = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            
            # 重塑为 (height, width, 4) - ARGB 格式
            img_data = img_data.reshape(height, width, 4)
            
            # 转换为 RGB (丢弃 Alpha 通道)
            img_data = img_data[:, :, 1:]

            img_data = cv2.resize(img_data, (640, 480), interpolation=cv2.INTER_AREA)
            
            # 无头模式：不显示窗口，直接返回图像
            return position, rotation, img_data
        else:
            # 交互模式：返回位置和旋转，窗口已显示
            return position, rotation


class Transformer:
    def __init__(self):
        self.visualizer = TransformVisualizer(space_size=2)

        self.left_initial = None
        self.right_initial = None
        self.initial_transform_set = False
        
        # 新增：存储前20帧数据
        self.left_frames = []
        self.right_frames = []
        self.frame_count = 0

    def _calculate_robust_average(self, frames):
        """计算鲁棒平均值，去掉最大值和最小值"""
        if len(frames) < 3:
            return frames[-1] if frames else None
            
        # 提取位置和旋转
        positions = [frame[:3, 3] for frame in frames]
        rotations = [frame[:3, :3] for frame in frames]
        
        # 计算位置的平均值（去掉最大和最小值）
        pos_array = np.array(positions)
        robust_pos = []
        for i in range(3):  # 对x, y, z分别处理
            values = pos_array[:, i]
            # 去掉最大值和最小值
            masked_values = np.delete(values, [np.argmax(values), np.argmin(values)])
            robust_pos.append(np.mean(masked_values))
        
        # 计算旋转的平均值（使用四元数平均）
        quats = [R.from_matrix(rot).as_quat() for rot in rotations]
        quat_array = np.array(quats)
        
        # 计算每个四元数与平均值的角度差
        avg_quat = np.mean(quat_array, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)  # 归一化
        
        angles = []
        for q in quats:
            dot = np.abs(np.dot(q, avg_quat))
            angle = 2 * np.arccos(np.clip(dot, -1.0, 1.0))
            angles.append(angle)
        
        # 去掉角度最大和最小的两个四元数
        angles = np.array(angles)
        indices_to_keep = np.argsort(angles)[1:-1]  # 去掉最大和最小
        
        robust_quats = quat_array[indices_to_keep]
        robust_avg_quat = np.mean(robust_quats, axis=0)
        robust_avg_quat = robust_avg_quat / np.linalg.norm(robust_avg_quat)
        
        # 构建变换矩阵
        result = np.eye(4)
        result[:3, 3] = robust_pos
        result[:3, :3] = R.from_quat(robust_avg_quat).as_matrix()
        
        return result

    def trans(self, position, rotation, name):
        quat = np.roll(rotation, -1)  # to [x, y, z, w]

        # 创建旋转矩阵
        rot = R.from_quat(quat)
        rot_matrix = rot.as_matrix()
        
        # 构造变换矩阵
        T_world = np.eye(4)
        T_world[:3, :3] = rot_matrix
        T_world[:3, 3] = position
        
        # 初始旋转校正：绕X轴旋转 -20度（roll）
        initial_rotation = xyzrpy2Mat(0, 0, 0, -(20.0 / 180.0 * math.pi), 0, 0)
        # 对齐旋转：绕X轴 -90度，绕Y轴 -90度
        alignment_rotation = xyzrpy2Mat(0, 0, 0, -90/180*math.pi, -90/180*math.pi, 0)
        rotate_matrix = np.dot(initial_rotation, alignment_rotation)
        
        # 平移变换：将采集到的pose数据变换到夹爪中心
        transform_matrix = xyzrpy2Mat(0.172, 0, -0.076, 0, 0, 0)
        result_mat = np.matmul(np.matmul(T_world, rotate_matrix), transform_matrix)
        
        # 区分设备
        is_left = name == PIKA_LEFT_NAME
        is_right = name == PIKA_RIGHT_NAME
        
        # 收集前20帧数据
        if not self.initial_transform_set:
            self.frame_count += 1
            
            if is_left:
                self.left_frames.append(result_mat.copy())
            elif is_right:
                self.right_frames.append(result_mat.copy())
            
            # 当收集到足够帧数时计算初始变换
            if len(self.left_frames) >= 20 and len(self.right_frames) >= 20:
                # # 使用第6帧到第20帧的数据
                # left_avg = self._calculate_robust_average(self.left_frames[5:20])
                # right_avg = self._calculate_robust_average(self.right_frames[5:20])
                left_avg = self.left_frames[19]
                right_avg = self.right_frames[19]
                
                if left_avg is not None and right_avg is not None:
                    # 计算中间位置
                    mid_pos = 0.5 * (left_avg[:3, 3] + right_avg[:3, 3])
                    
                    # 计算中间旋转（使用Slerp）
                    # r_left = R.from_matrix(left_avg[:3, :3])
                    # r_right = R.from_matrix(right_avg[:3, :3])
                    # quat_left = r_left.as_quat()
                    # quat_right = r_right.as_quat()
                    # quat_mid = quat_slerp(quat_left, quat_right, 0.5)
                    # mid_rot = R.from_quat(quat_mid)
                    # mid_rot_matrix = mid_rot.as_matrix()
                    
                    # 构造中间变换矩阵
                    mid_T = np.eye(4)
                    # mid_T[:3, :3] = mid_rot_matrix
                    mid_T[:3, :3] = left_avg[:3, :3]
                    mid_T[:3, 3] = mid_pos
                    
                    self.visualizer.set_initial_transform(mid_T)
                    self.initial_transform_set = True
                    logger.info("Initial transform set using frames 6-20 average (with outlier removal).")
        
        # 更新对应设备的变换
        if self.initial_transform_set:
            device_id = 'left' if is_left else 'right' if is_right else None
            if device_id:
                return self.visualizer.update_transform(device_id, result_mat)
        else:
            return None
    
    def close(self):
        plt.close(self.visualizer.fig)