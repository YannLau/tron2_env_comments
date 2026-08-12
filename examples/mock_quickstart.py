"""Run a software-only MotionController example without connecting to a robot.

本示例演示：**不连接任何真实机器人**，只用纯软件（一个"内存里的假机器人"）
就能跑通 MotionController 的完整控制流程。它是新手了解本仓库的最佳起点。

建议阅读顺序：
  1. 模块顶部 / JointIndex —— 先搞懂"18 维状态"和"16 维控制指令"分别是什么。
  2. RecordingTransport —— 一个"假"的机器人连接，把我们发出的指令记录下来。
  3. main() —— 整个控制流程的主干（创建 -> 启动 -> 下发目标 -> 验证 -> 断开）。

运行方式：
    python examples/mock_quickstart.py
"""

from __future__ import annotations

import threading
import time

import numpy as np

from tron2_env.joints import JointIndex
from tron2_env.motion import MotionController


class RecordingTransport:
    """In-memory RobotTransport used only by this no-hardware example.

    这是一个"内存版"的 RobotTransport：它不连接任何真实机器人，而是把我们
    通过 send_joint_cmd() 发出的每一帧关节指令存进内存列表，供事后检查/断言。

    为什么要写这样一个类？
    - 整个 tron2_env 的控制流程只依赖一个"抽象接口" RobotTransport
      （见 src/tron2_env/transport/base.py，它是一个 Protocol，即鸭子类型）。
    - 只要实现了 send_joint_cmd / get_joint_state / ... 这几个方法，
      MotionController 就能照常工作，完全不需要真实硬件。
    - 所以这个假 transport 既能做单元测试，也能让新手在无硬件环境下跑通全流程。

    维度说明（常量定义见 src/tron2_env/joints.py 的 JointIndex）：
    - 状态 STATE_DIM = 18：[左臂(7), 左夹爪(1), 右臂(7), 右夹爪(1), 头(2)]
    - 控制指令 SERVOJ_DIM = 16：[左臂(7), 右臂(7), 头(2)]
      注意：夹爪不在 16 维控制指令里，夹爪由 set_gripper() 单独控制。
    """

    def __init__(self) -> None:
        self._connected = True
        self._lock = threading.Lock()
        self._sent: list[np.ndarray] = []
        # "机器人当前状态"，假装一直停在初始位置（全部关节角为 0）。
        self._state = np.zeros(JointIndex.STATE_DIM, dtype=np.float64)

    def send_joint_cmd(self, q: np.ndarray) -> None:
        # MotionController 的发布线程会反复调用这个方法，把一帧关节指令"发给机器人"。
        # 真实实现（WebsocketTransport）会把它序列化成 JSON、走 websocket 发到真机；
        # 这里只是复制一份存进内存列表，方便最后验证我们确实发出了指令。
        with self._lock:
            self._sent.append(np.asarray(q, dtype=np.float64).copy())

    def get_joint_state(self, timeout: float = 1.0) -> dict:
        # 读取机器人当前的 18 维状态。MotionController.start() 会调用它一次，
        # 用当前实际关节角给插值器"播种"起点 —— 这样从启动那刻起就不会突然跳变。
        # 这里我们假装机器人一直停在初始位置，时间戳用当前毫秒数。
        del timeout
        return {"timestamp": int(time.time() * 1000), "states": self._state.tolist()}

    def get_head_position(self) -> np.ndarray:
        # 返回头部 [pitch, yaw] 两个关节角。直接从 state 里切出头部那一段。
        return self._state[JointIndex.HEAD].copy()

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        # 设置左右夹爪开合度（0~100）。假实现里什么都不做。
        del left_opening, right_opening

    def wait_until_reached(
        self,
        target_joints,
        tolerance: float = 0.05,
        timeout: float = 10.0,
    ) -> bool:
        # 阻塞式轮询：直到机械臂关节进入 target_joints 的 tolerance 范围才返回。
        # 假实现里认为"永远已经到位"，直接返回 True。
        del target_joints, tolerance, timeout
        return True

    def disconnect(self) -> None:
        # 断开连接。假实现只需把连接标记置为 False。
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        # 支持 with 语法：进入时返回自身。
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # with 块结束时自动调用 disconnect()，确保连接被释放。
        del exc_type, exc_val, exc_tb
        self.disconnect()

    def sent_frames(self) -> list[np.ndarray]:
        # 自定义辅助方法（不属于 RobotTransport 接口）：返回所有"发出去"的指令帧，
        # 供测试断言使用。注意加了锁，因为发布线程和主线程会同时访问 _sent。
        with self._lock:
            return list(self._sent)


def main() -> None:
    # 1. 建一个"假"机器人连接。
    transport = RecordingTransport()

    # 2. 创建 MotionController —— 真正的核心对象（见 src/tron2_env/motion/controller.py）。
    #    - transport:     底层传输层。这里用假的；真实场景换成 WebsocketTransport。
    #    - publish_rate:  后台发布线程每秒向机器人发送多少帧指令（这里 100 Hz）。
    #    - eta_default:   command_joints() 未显式指定 eta 时使用的默认值；
    #                     eta = "到达目标需要的时长"（秒）。0.02s 即 20ms。
    #    它的内部结构（理解后事半功倍）：
    #       a. 插值器 LinearInterpolator 负责把"当前关节角 → 目标关节角"平滑过渡，
    #          避免瞬间跳到目标、导致机器人猛冲；
    #       b. 一个后台 daemon 线程按 publish_rate 的节奏，反复取插值器的当前值，
    #          通过 transport.send_joint_cmd() 发给机器人；
    #       c. 机器人侧自带 PD 控制，只要持续收到指令帧，就会稳稳停在最新目标附近。
    controller = MotionController(transport=transport, publish_rate=100.0, eta_default=0.02)

    # 3. 生成一个 16 维的目标关节角（单位：弧度）。
    #    np.linspace(-0.1, 0.1, 16) 表示从 -0.1 到 0.1 均匀取 16 个点，
    #    即按 [左臂7角, 右臂7角, 头2角] 的顺序拼成一条目标指令。
    target = np.linspace(-0.1, 0.1, JointIndex.SERVOJ_DIM)

    # 4. 启动控制器：
    #    - 先读一次机器人当前状态，把插值器起点设成"当前实际关节角"（防止起跳）；
    #    - 然后启动后台发布线程。
    controller.start()
    try:
        # 5. 重新定向到目标 —— 注意这是"非阻塞"的！
        #    command_joints() 只做一件事：把插值器的"目的地"改掉，然后立刻返回。
        #    真正的平滑运动，由后台发布线程在接下来的 100Hz 循环里逐步完成。
        controller.command_joints(target)

        # 6. 主线程睡 0.08 秒，给后台发布线程一点时间多发布几帧。
        #    100 Hz 下大约会发出 8 帧左右（前几帧还是起始位置，随后逐渐逼近目标）。
        time.sleep(0.08)
    finally:
        # 7. finally 保证无论上面是否抛异常，都会断开：
        #    停止后台线程并释放连接。
        controller.disconnect()

    # 8. 验证一：确实有指令帧被发出过（空列表说明流程没跑起来）。
    frames = transport.sent_frames()
    if not frames:
        raise RuntimeError("mock controller did not publish any frames")

    # 9. 验证二：最后一帧指令应当（在容差内）等于目标关节角。
    #    因为 eta=0.02s，0.08s 的等待已远超到达时间，插值器应已到达目标。
    #    np.testing.assert_allclose 是 numpy 自带的断言，不等会抛异常。
    np.testing.assert_allclose(frames[-1], target, atol=1e-6)
    print(f"Mock transport published {len(frames)} frames; no robot connection was opened.")


if __name__ == "__main__":
    # 只有"直接运行本文件"时才会执行 main()；
    # 如果被其他模块 import，则不会执行 —— 这是 Python 的标准约定。
    main()
