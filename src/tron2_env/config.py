"""机器人级（Robot-level）配置 —— :class:`Tron2Config` 数据类。

本文件是连接 TRON2 机器人时的"总开关"：环境（``env.py``）、运动控制器
（``motion/controller.py``）和 WebSocket 传输层（``transport/websocket.py``）
共用同一份 :class:`Tron2Config`，决定"连哪里、连上后先摆成什么姿态、状态
怎么同步"。

它只负责**声明**配置，本身不做任何网络 I/O 或机器人操作。

四大类配置（也对应下方字段的注释分组）：

=============  ==============================================================
类别           字段
=============  ==============================================================
连接参数       ``robot_ip`` / ``port`` —— 机器人控制盒的地址和端口
上电位姿       ``init_joints`` / ``init_head`` —— 连上后先把机器人摆到哪
安全参数       ``init_ee_z_min`` —— 末端高度下限，低于它时强制绕行路径
传输层调参     ``state_queue_maxlen`` / ``polling_rate`` / ``connection_timeout``
=============  ==============================================================

新手快速上手：绝大多数场景只需改 ``robot_ip`` 一项，其余字段都有安全默认值::

    from tron2_env.config import Tron2Config
    from tron2_env.env import EnvConfig

    # 1. 连接真机：把 "ROBOT_IP" 换成控制盒实际 IP（默认值只是占位符，连不上真机）
    robot_config = Tron2Config(robot_ip="192.168.1.10")

    # 2. 全部用默认值（适合只跑逻辑、不连真机的测试）
    robot_config = Tron2Config()

    # 3. 自定义上电位姿：14 维双臂关节角 + 2 维头部关节角（单位都是弧度）
    robot_config = Tron2Config(
        init_joints=[0.0] * 14,       # [左臂 7 关节, 右臂 7 关节]，不含夹爪
        init_head=[0.0, 0.0],         # [头部 pitch, 头部 yaw]
    )

    env_config = EnvConfig(robot_config=robot_config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tron2_env.joints import JointIndex


@dataclass
class Tron2Config:
    """TRON2 机器人的连接参数与上电初始化参数。

    这是一个 ``@dataclass``：所有字段都有默认值，因此
    ``Tron2Config()`` 即可构造；想改哪项就用关键字参数覆盖哪项。

    构造时（``__post_init__``）会自动校验 ``init_joints`` / ``init_head``
    的维度，维度不对会立刻抛出 ``ValueError``，避免错误配置跑到真机上。

    典型使用方式：::

        config = Tron2Config(robot_ip="192.168.1.10")

    注意：``None`` 表示"该字段不生效"（例如 ``init_joints=None`` 表示
    连接后不发送初始关节角指令），这一点对三个 ``Optional`` 字段通用。
    """

    # ------------------------------------------------------------------
    # 第一组：WebSocket 连接参数 —— 机器人在哪、从哪个端口进
    # ------------------------------------------------------------------

    # 机器人控制盒的 IP 地址。
    # 默认值 "ROBOT_IP" 只是一个占位符：不改成真机 IP 就无法连接。
    # 在传输层 websocket.py 中被用作 ws://<robot_ip>:<port> 的地址。
    robot_ip: str = "ROBOT_IP"

    # WebSocket 服务端口。一般机器人出厂配置为 5000，除非控制盒改了端口，
    # 否则无需动它。
    port: int = 5000

    # ------------------------------------------------------------------
    # 第二组：上电位姿（bring-up pose）—— 连接成功后先把机器人摆成什么姿态
    # ------------------------------------------------------------------

    # 连接成功后双臂要移动到的目标关节角，14 维 = [左臂 7 关节, 右臂 7 关节]。
    #   * 维度来自 JointIndex.MOVEJ_DIM（ARM_DIM * 2 = 14），不含夹爪；
    #   * 单位：弧度（rad），不是角度；
    #   * None（默认）表示跳过这一步，保持上电后的当前位置不动。
    # 实际执行在 transport/websocket.py 的初始化流程里：先读当前状态，
    # 若末端过低还会先走安全绕行（见 init_ee_z_min），然后以
    # movej(..., move_time=2.0) 下发这条轨迹。
    init_joints: Optional[List[float]] = None      # 14-dim (arm-only)

    # 连接成功后头部（head）要移动到的目标关节角，2 维 = [pitch, yaw]。
    #   * 维度来自 JointIndex.HEAD_DIM = 2；
    #   * 单位：弧度（rad）；
    #   * None（默认）表示不发送头部指令，头保持不动。
    # 在初始化流程中排在 init_joints 之后执行：move_head(..., move_time=1.0)。
    init_head: Optional[List[float]] = None        # 2-dim

    # 上电位姿的安全阈值（单位：米）。
    # 初始化时若双臂末端（end-effector）中**任一**的 Z 坐标低于该值，
    # 说明机械臂初始位置太低，直接走直线（MoveJ）可能撞到台面——
    # 此时初始化流程会改为"经由第二关节绕行"的更安全路径。
    #   * None 表示禁用这条保护规则；
    #   * 默认 -0.6 米是经验安全值，一般不需要改。
    init_ee_z_min: Optional[float] = -0.6          # route via second joint if any EE z is below this

    # ------------------------------------------------------------------
    # 第三组：WebSocket 传输层内部调参 —— 状态怎么同步、连接等多久
    # ------------------------------------------------------------------

    # 状态缓冲队列的最大长度。
    # 传输层用 deque(maxlen=state_queue_maxlen) 分别缓存最近收到的
    # 关节状态（joint_state_queue）和末端位姿（ee_pose_queue）：
    # 满队时最旧的一条会被自动丢弃，保证读到的总是"最新 N 帧"。
    # 拉大能容忍偶尔的网络抖动，但会增加端到端时延，7 是调试后的经验值。
    state_queue_maxlen: int = 7

    # 状态轮询线程的请求频率（单位：Hz）。
    # 传输层会启动一个后台线程，以 1/polling_rate 秒为周期主动向机器人
    # 索取当前状态。200 Hz 即每 5 ms 请求一次。
    # 注意：这是"发送请求的频率"，实际状态刷新还受机器人响应速度和
    # 网络往返延迟限制；调高不会突破真机硬件的上限。
    polling_rate: float = 200.0

    # WebSocket 建立连接的超时时间（单位：秒）。
    # 超过该时间还没连上控制盒就报错退出，而不是无限等待。
    # 5 秒对局域网内连接通常足够；若走较慢的链路可适当调大。
    connection_timeout: float = 5.0

    def __post_init__(self) -> None:
        """dataclass 构造完成后的自动校验（由 ``@dataclass`` 自动调用）。

        校验规则：
          * ``init_joints`` 非 None 时必须是 14 维（JointIndex.MOVEJ_DIM）
            = 左臂 7 关节 + 右臂 7 关节；
          * ``init_head`` 非 None 时必须是 2 维（JointIndex.HEAD_DIM）
            = 头部 pitch + yaw。

        维度写错会立刻抛 ``ValueError`` 并附带"期望 vs 实际"的提示，
        把低级配置错误挡在连接真机之前。
        """
        if self.init_joints is not None and len(self.init_joints) != JointIndex.MOVEJ_DIM:
            raise ValueError(
                f"init_joints must have {JointIndex.MOVEJ_DIM} elements, got {len(self.init_joints)}"
            )
        if self.init_head is not None and len(self.init_head) != JointIndex.HEAD_DIM:
            raise ValueError(
                f"init_head must have {JointIndex.HEAD_DIM} elements, got {len(self.init_head)}"
            )
