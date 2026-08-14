from tron2_env.transport.websocket import WebsocketTransport
from tron2_env.config import Tron2Config
import time
import argparse



def main() -> None:

    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Send one movej joint command to the TRON2 robot and exit."
    )
    parser.add_argument("--a", default=1, type=int)
    a = 1

    name = parser.parse_args()

    a = name.a

    config = Tron2Config(robot_ip="10.192.1.2",port="5000")

    DEFAULT_INIT_HEAD = [1.0467, -0.0139998]
    DEFAULT_INIT_JOINTS = [
        0.026899,    # 左臂关节1
        0.2612,      # 左臂关节2
        -0.02709991, # 左臂关节3
        -1.5477003,  # 左臂关节4
        0.265,       # 左臂关节5
        0.0180999,   # 左臂关节6
        -0.0614999,  # 左臂关节7
        0.008999,    # 右臂关节1
        -0.269,      # 右臂关节2
        0.02069998,  # 右臂关节3
        -1.5567001,  # 右臂关节4
        -0.254,      # 右臂关节5
        -0.02309972, # 右臂关节6
        0.06469989,  # 右臂关节7
    ]

    up_head = [-0.62,-0.0139998]
    amazing = [0.000999913, -0.00449967, 1.482, -1.57, 0.0036, 0.00289989, -0.00160009, 0.0415001, 0.1279, -1.4808, -1.57, -0.00739986, 0.0151, -0.0624998,]


    if a == 1:
        with WebsocketTransport(config=config) as transport:
            transport.move_head(DEFAULT_INIT_HEAD,move_time=3)
            time.sleep(3)
            transport.movej(DEFAULT_INIT_JOINTS,move_time=2)
    elif a == 2:
        with WebsocketTransport(config=config) as transport:
            transport.move_head(up_head, move_time=3)
            time.sleep(3)
            transport.movej(amazing, move_time=2)     

# ── 程序入口 ──
if __name__ == "__main__":
    main()