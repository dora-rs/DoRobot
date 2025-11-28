import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import os
import pyarrow as pa

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from dora import Node
from pose_utils import xyzQuaternion2matrix, xyzrpy2Mat, matrixToXYZQuaternion

logger = logging.getLogger(__name__)
RIGHT_SN = os.getenv("RIGHT_SN")
LEFT_SN = os.getenv("LEFT_SN")

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
    def __init__(self, axis_length=0.1, space_size=2):
        self.axis_length = axis_length
        self.space_size = space_size
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(azim=-160)
        self._setup_plot()
        
        # 存储多个设备的变换
        self.transforms = {}  # {device_id: transform_matrix}
        self.trajectory_points = {}  # {device_id: [points]}
        self.T_zero = None
        self.device_colors = {'left': 'cyan', 'right': 'magenta'}  # 设备显示颜色

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
        plt.ion()
        plt.show()

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
                self.plot_trajectory(device_id, color=color)
        
        # 刷新画布
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

def main():
    node = Node()
    visualizer = TransformVisualizer(space_size=2)
    
    left_initial = None
    right_initial = None
    initial_transform_set = False
    
    try:
        for event in node:
            if event["type"] == "INPUT" and event["id"] == "pose":
                data = event["value"][0]
                serial_number = data.get("serial_number").as_py()
                
                # 解析位置和旋转
                position = np.array(data["position"].as_py())
                rotation = data["rotation"].as_py()
                quat = np.array(rotation[1:] + rotation[:1])  # [x, y, z, w]
                
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
                is_left = serial_number == LEFT_SN
                is_right = serial_number == RIGHT_SN
                
                # 初始化基准变换
                if is_left or is_right:
                    if is_left and left_initial is None:
                        left_initial = result_mat.copy()
                    elif is_right and right_initial is None:
                        right_initial = result_mat.copy()
                    
                    if left_initial is not None and right_initial is not None and not initial_transform_set:
                        # 计算中间位置
                        mid_pos = 0.5 * (left_initial[:3, 3] + right_initial[:3, 3])
                        
                        # 计算中间旋转（使用Slerp）
                        r_left = R.from_matrix(left_initial[:3, :3])
                        r_right = R.from_matrix(right_initial[:3, :3])
                        quat_left = r_left.as_quat()
                        quat_right = r_right.as_quat()
                        quat_mid = quat_slerp(quat_left, quat_right, 0.5)
                        mid_rot = R.from_quat(quat_mid)
                        mid_rot_matrix = mid_rot.as_matrix()
                        
                        # 构造中间变换矩阵
                        mid_T = np.eye(4)
                        mid_T[:3, :3] = mid_rot_matrix
                        mid_T[:3, 3] = mid_pos
                        
                        visualizer.set_initial_transform(mid_T)
                        initial_transform_set = True
                        logger.info("Initial transform set as the midpoint between LEFT and RIGHT devices.")
                
                # 更新对应设备的变换
                if initial_transform_set:
                    device_id = 'left' if is_left else 'right' if is_right else None
                    if device_id:
                        visualizer.update_transform(device_id, result_mat)
                        # node.send_output(f"{device_id}_pose", pa.array(result_mat.ravel()))
            
            elif event["type"] == "STOP":
                break
                
    except KeyboardInterrupt:
        logger.info("\nExiting dora_pika_visual...")
    except Exception as e:
        logger.exception("Dora error: %s", e)
    finally:
        plt.close(visualizer.fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()