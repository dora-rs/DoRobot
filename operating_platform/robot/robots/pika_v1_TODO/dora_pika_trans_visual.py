import numpy as np
import matplotlib.pyplot as plt
import logging
import math

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from dora import Node
from pose_utils import xyzQuaternion2matrix, xyzrpy2Mat, matrixToXYZQuaternion


logger = logging.getLogger(__name__)

class TransformVisualizer:
    def __init__(self, axis_length=0.1, space_size=2):
        self.axis_length = axis_length
        self.space_size = space_size
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(azim=-160)
        self._setup_plot()
        self.current_transform = None  # 存储当前最新变换
        self.trajectory_points = []
        self.T_zero = None
        
    def _setup_plot(self):
        """初始化3D坐标系和图形设置"""
        base_T = np.eye(4)
        self.plot_transform(base_T, axis_length=2.0, label="Base Frame")
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("6D Visual")
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.legend()
        
        limits = [-self.space_size, self.space_size]
        self.ax.set_xlim(limits)
        self.ax.set_ylim(limits)
        self.ax.set_zlim(limits)

        plt.tight_layout()
        plt.ion()
        plt.show()
        
    def plot_transform(self, T, axis_length=None, label=None):
        """绘制单个坐标系变换"""
        axis_length = axis_length or self.axis_length
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * axis_length
        y_axis = origin + T[:3, 1] * axis_length
        z_axis = origin + T[:3, 2] * axis_length
        
        # 绘制轴线
        self.ax.plot([origin[0], x_axis[0]], 
                    [origin[1], x_axis[1]], 
                    [origin[2], x_axis[2]], c='r', linewidth=2)
        self.ax.plot([origin[0], y_axis[0]], 
                    [origin[1], y_axis[1]], 
                    [origin[2], y_axis[2]], c='g', linewidth=2)
        self.ax.plot([origin[0], z_axis[0]], 
                    [origin[1], z_axis[1]], 
                    [origin[2], z_axis[2]], c='b', linewidth=2)
        
        if label:
            self.ax.text(*origin, label, color='k', fontsize=6)
            
    def plot_trajectory(self, color='gray', label='Trajectory'):
        """绘制轨迹"""
        if not self.trajectory_points:
            return
            
        points = np.array(self.trajectory_points)
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                    color=color, label=label)
    
    def update_visualization(self):
        """更新可视化界面"""
        # 清除旧元素
        self.ax.cla()
        self._setup_plot()
        
        # 绘制当前变换（如果有）
        if self.current_transform is not None:
            self.plot_transform(self.current_transform, label="Current Pose")
            
        # 更新轨迹
        self.plot_trajectory()
        
        # 刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_transform(self, T_world):
        """更新当前变换并刷新可视化"""
        # 设置初始参考系
        if self.T_zero is None:
            self.T_zero = T_world
            
        # 计算相对变换
        T_relative = np.linalg.inv(self.T_zero) @ T_world
        # T_relative = T_world


        # 更新当前变换
        self.current_transform = T_relative
        
        # 记录轨迹点
        self.trajectory_points.append(T_relative[:3, 3])
        
        # 刷新可视化
        self.update_visualization()


def main():
    node = Node()
    visualizer = TransformVisualizer(space_size=2)
    
    try:
        for event in node:
            if event["type"] == "INPUT" and event["id"] == "pose":
                data = event["value"][0]
                
                # 解析位置和旋转
                position = np.array(data["position"].as_py())
                rotation = data["rotation"].as_py()
                
                # 构造四元数 (x, y, z, w)
                quat = np.array(rotation[1:] + rotation[:1])
                
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
                # 合并旋转矩阵
                rotate_matrix = np.dot(initial_rotation,alignment_rotation)
                # 应用平移变换 - 将采集到的pose数据变换到夹爪中心
                transform_matrix = xyzrpy2Mat(0.172, 0, -0.076, 0, 0, 0)
                # 计算最终变换矩阵
                result_mat = np.matmul(np.matmul(T_world, rotate_matrix), transform_matrix)

                # T_world = flip_matrix @ T_world
                # T_world = T_world @ 
                
                logger.info(f"Received transform: {result_mat}")
                visualizer.update_transform(result_mat)
                
            elif event["type"] == "STOP":
                break
                
    except KeyboardInterrupt:
        logger.info("\nExiting dora_vive_visual...")
    except Exception as e:
        logger.exception("Dora error: %s", e)
    finally:
        plt.close(visualizer.fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()