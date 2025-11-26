import os
import pyarrow as pa
from dora import Node
from Robotic_Arm.rm_robot_interface import *


id = os.getenv("ARM_ID", "arm_right")       # ARM_ID: Arm identifier; defaults to "arm_right" if not set
ip = os.getenv("ARM_IP", "192.168.1.19")    # ARM_IP: Connection IP address; defaults to "192.168.1.18" if not set
port = int(os.getenv("ARM_PORT", "8080"))   # ARM_PORT: Connection port number; defaults to 8080 if not set

start_pose_str = os.getenv("START_POSE", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
joint_p_limit_str = os.getenv("JOINT_P_LIMIT", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
joint_n_limit_str = os.getenv("JOINT_N_LIMIT", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
start_pose = [float(x) for x in start_pose_str.split(',')]
joint_p_limit = [float(x) for x in joint_p_limit_str.split(',')]
joint_n_limit = [float(x) for x in joint_n_limit_str.split(',')]

node = Node()


class RealmanArm:
    def __init__(self):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        
        handle = self.arm.rm_create_robot_arm(ip, port)
        print("Arm ID: ", id)
        print("Arm handle ID: ", handle.id)
        print(f"Arm Connected On: {ip}:{port}")
        software_info = self.arm.rm_get_arm_software_info()
        if software_info[0] == 0:
            print("\n================== Arm Software Information ==================")
            print("Arm Model: ", software_info[1]['product_version'])
            print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
            print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
            print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
            print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
            print("==============================================================\n")
        else:
            print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")

        self.arm.rm_set_tool_voltage(3)#设置末端工具接口电压为24v
        self.arm.rm_set_modbus_mode(1, 115200, 5) #打开modbus模式
        self.peripheral = rm_peripheral_read_write_params_t(1, 40000, 1, 1)#配置串口参数
        self.arm.rm_write_single_register(self.peripheral, 100)#初始化夹爪为打开状态
        
        self.is_connected = True
        self.logs = {}
    
    def movej_cmd(self, joint):
        clipped_joint = max(joint_n_limit[:7], min(joint_p_limit[:7], joint[:7]))
        self.arm.rm_movej(clipped_joint, 30, 0, 0, 0)
    
    def movej_canfd(self, joint):
        clipped_joint = max(joint_n_limit[:7], min(joint_p_limit[:7], joint[:7]))
        self.arm.rm_movej_canfd(clipped_joint, True, 0, 1, 50)

    def write_single_register(self, gripper):
        clipped_gripper = max(0, min(100, gripper))
        self.arm.rm_write_single_register(self.peripheral, clipped_gripper)

    def read_joint_degree(self):
        _num, joint_read = self.arm.rm_get_joint_degree()
        return joint_read
    
    def stop(self):
        self.arm.rm_set_arm_stop()
    
    def disconnect(self):
        self.arm.rm_close_modbus_mode(1)
        self.is_connected = False


def main():
    main_arm = RealmanArm(ip, port, start_pose, joint_p_limit, joint_n_limit)

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            if event["id"] == "movej-cmd":
                joint = event["value"].to_pylist()
                main_arm.movej_cmd(joint)
                
            if event["id"] == "movej":
                joint = event["value"].to_pylist()
                main_arm.movej_canfd(joint)

            if event["id"] == "gripper":
                gripper = event["value"]
                main_arm.write_single_register(gripper)

            if event["id"] == "read-joint":
                read_joint = main_arm.read_joint_degree()
                node.send_output("read-joint", pa.array(read_joint))

            if event["id"] == "stop":
                main_arm.stop()
        
        elif event_type == "ERROR":
            print("Event Error:" + event["error"])
    
    main_arm.disconnect()


if __name__ == "__main__":
    main()
