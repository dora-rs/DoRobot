"""
Dual Device 6D Transformation Visualizer

This module visualizes 3D transformations from two devices (left and right) using
matplotlib 3D plotting with SLERP interpolation for smooth rotations.
"""

import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from dora import Node
from pose_utils import xyzQuaternion2matrix, xyzrpy2Mat, matrixToXYZQuaternion

logger = logging.getLogger(__name__)
RIGHT_SN = os.getenv("RIGHT_SN")
LEFT_SN = os.getenv("LEFT_SN")


def quat_slerp(q0, q1, t):
    """
    Manually implement quaternion SLERP interpolation.
    
    Args:
        q0 (np.ndarray): First quaternion in [x, y, z, w] format.
        q1 (np.ndarray): Second quaternion in [x, y, z, w] format.
        t (float): Interpolation parameter in [0, 1].
        
    Returns:
        np.ndarray: Interpolated quaternion.
    """
    dot = np.dot(q0, q1)
    
    # Ensure shortest path interpolation
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    # Linear interpolation for small angles
    if dot > 0.9995:
        return (1 - t) * q0 + t * q1
    
    # Spherical interpolation
    theta = np.arccos(dot) * t
    sin_theta = np.sin(theta)
    sin_full = np.sin(theta / t)
    
    return (np.sin(theta - theta / t) * q0 + sin_theta * q1) / sin_full


class TransformVisualizer:
    """
    Visualizer for 3D transformations with trajectory tracking.
    
    Attributes:
        axis_length: Length of coordinate frame axes.
        space_size: Size of 3D visualization space.
        transforms: Dictionary storing current transformations.
        trajectory_points: Dictionary storing trajectory points.
        device_colors: Color configuration for left/right devices.
    """
    
    def __init__(self, axis_length=0.1, space_size=2):
        """Initialize 3D visualization environment."""
        self.axis_length = axis_length
        self.space_size = space_size
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(azim=-160)
        
        # Initialize transform data
        self.transforms = {}  # {device_id: transform_matrix}
        self.trajectory_points = {}  # {device_id: [points]}
        
        # Device color configuration
        self.device_colors = {
            'left': {'x': 'cyan', 'y': 'lime', 'z': 'deepskyblue'},
            'right': {'x': 'magenta', 'y': 'yellow', 'z': 'orange'}
        }
        
        self.T_zero = None
        self._setup_plot()

    def _setup_plot(self):
        """Initialize 3D coordinate system and visualization settings."""
        # Plot base frame
        base_T = np.eye(4)
        self.plot_transform(base_T, axis_length=2.0, label="Base Frame")
        
        # Configure plot appearance
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Dual Device 6D Visualization")
        self.ax.set_box_aspect([1, 1, 1])
        
        # Set coordinate limits
        limits = [-self.space_size, self.space_size]
        self.ax.set_xlim(limits)
        self.ax.set_ylim(limits)
        self.ax.set_zlim(limits)
        
        # Add legend
        self.ax.legend([
            plt.Line2D([0], [0], color='cyan', lw=2),
            plt.Line2D([0], [0], color='magenta', lw=2),
            plt.Line2D([0], [0], color='black', lw=2)
        ], ['Left Device', 'Right Device', 'Base Frame'])
        
        plt.tight_layout()
        plt.ion()
        plt.show()

    def plot_transform(self, T, axis_length=None, label=None, device_id='default'):
        """
        Plot a coordinate frame transformation.
        
        Args:
            T: 4x4 transformation matrix.
            axis_length: Length of axis lines.
            label: Text label for the frame.
            device_id: Identifier for device-specific colors.
        """
        axis_length = axis_length or self.axis_length
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * axis_length
        y_axis = origin + T[:3, 1] * axis_length
        z_axis = origin + T[:3, 2] * axis_length
        
        # Get device-specific colors
        colors = self.device_colors.get(device_id, {'x': 'r', 'y': 'g', 'z': 'b'})
        
        # Plot each axis
        self.ax.plot([origin[0], x_axis[0]], 
                    [origin[1], x_axis[1]], 
                    [origin[2], x_axis[2]], 
                    c=colors['x'], linewidth=2, alpha=0.8)
        
        self.ax.plot([origin[0], y_axis[0]], 
                    [origin[1], y_axis[1]], 
                    [origin[2], y_axis[2]], 
                    c=colors['y'], linewidth=2, alpha=0.8)
        
        self.ax.plot([origin[0], z_axis[0]], 
                    [origin[1], z_axis[1]], 
                    [origin[2], z_axis[2]], 
                    c=colors['z'], linewidth=2, alpha=0.8)
        
        if label:
            self.ax.text(*origin, label, color='k', fontsize=8)

    def plot_trajectory(self, device_id, color='gray'):
        """
        Plot device trajectory.
        
        Args:
            device_id: Identifier for the device.
            color: Color for trajectory line.
        """
        if device_id not in self.trajectory_points or not self.trajectory_points[device_id]:
            return
            
        points = np.array(self.trajectory_points[device_id])
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                    color=color, linewidth=1, alpha=0.5)

    def update_visualization(self):
        """Update the visualization with current transformations."""
        self.ax.cla()
        self._setup_plot()
        
        # Redraw all transforms and trajectories
        for device_id, transform in self.transforms.items():
            if transform is not None:
                # Get device-specific color configuration
                device_color = self.device_colors.get(
                    device_id, {'x': 'r', 'y': 'g', 'z': 'b'}
                )
                
                # Plot transform with device color
                self.plot_transform(
                    transform, 
                    label=f"{device_id.capitalize()} Pose",
                    device_id=device_id
                )
                
                # Plot trajectory with x-axis color
                trajectory_color = device_color['x']
                self.plot_trajectory(device_id, color=trajectory_color)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_initial_transform(self, T_initial):
        """
        Set initial transform as reference.
        
        Args:
            T_initial: 4x4 transformation matrix.
        """
        self.T_zero = T_initial.copy()

    def update_transform(self, device_id, T_world):
        """
        Update device transformation.
        
        Args:
            device_id: Identifier for the device.
            T_world: New world transformation matrix.
        """
        if self.T_zero is None:
            logger.warning("T_zero not set. Cannot update transform.")
            return
            
        # Calculate relative transform
        T_relative = np.linalg.inv(self.T_zero) @ T_world
        
        # Update transform data
        self.transforms[device_id] = T_relative
        
        # Record trajectory point
        if device_id not in self.trajectory_points:
            self.trajectory_points[device_id] = []
        self.trajectory_points[device_id].append(T_relative[:3, 3])
        
        # Update visualization
        self.update_visualization()


def main():
    """Main function to run the visualization node."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting dual device visualization node")
    
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
            
            elif event["type"] == "STOP":
                break
                
    except KeyboardInterrupt:
        logger.info("\nExiting visualization node...")
    except Exception as e:
        logger.exception("Node error: %s", e)
    finally:
        plt.close(visualizer.fig)


if __name__ == "__main__":
    main()