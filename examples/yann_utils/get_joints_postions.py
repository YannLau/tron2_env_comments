from tron2_env.config import Tron2Config

from tron2_env.transport.websocket import WebsocketTransport

from datetime import datetime


time = datetime.now()

if __name__=="__main__":
    config = Tron2Config(robot_ip="10.192.1.2",port="5000")
    with WebsocketTransport(config=config) as transport:
        current = transport.get_joint_state(timeout=50)['states']
        from tron2_env.joints import JointIndex
        arm = current[JointIndex.LEFT_ARM] + current[JointIndex.RIGHT_ARM]
        print(f"Tron2 Arm (左臂 + 右臂) = {arm}")
        head = current[JointIndex.HEAD]
        print(f"Tron2 Head = {head}")
        # 处理"夹爪状态响应"。机器人回报的开合度是 0~100 的百分比，
        # 这里 /100 归一化成 0~1（和关节角同一数量级，方便一起处理）。
        left_gripper = current[JointIndex.LEFT_GRIPPER]
        right_gripper = current[JointIndex.RIGHT_GRIPPER]
        print(f"Tron2 left gripper = {left_gripper}")
        print(f"Tron2 right gripper = {right_gripper}")
        
        print(f"{time}获取关节夹爪状态结束")
        