"""
Tron2 Real Robot Environment

提供 Tron2 机器人的环境封装:
- 机器人控制委托给 ``MotionController`` (transport + interpolator + publish loop)
- 观测来源切换:bridge WebSocket(图像+关节由 bridge 对齐)或 legacy(RealSense 直连)
- ``step(action)`` 非阻塞:把 16-dim 目标交给 publish 线程,后台以 ``publish_rate``
  (默认 300 Hz)持续 send_joint_cmd,在 ``eta = 1/fps`` 时间内平滑过渡

============================================================================
给新手的第一课：这个文件在整条系统里是什么位置？
============================================================================
Tron2 是一台双臂机器人。要"喂给神经网络训练"或"在线跑策略"，需要两样东西：
  1) 动作的出口  —— 把策略算出的关节角发给机器人（控制）
  2) 观测的入口  —— 从机器人/相机读出状态和图像（感知）

本文件就是包住这两样东西的"环境"层（很标准的 RL Gym 风格接口）：
    ┌─────────────┐   step(action)    ┌──────────────────┐   send_joint_cmd   ┌────────┐
    │  策略/模型    │ ───────────────► │  Tron2Env(本文件) │ ─────────────────► │ 机器人   │
    │ (policy)    │ ◄─────────────── │                  │ ◄───────────────── │(robot) │
    └─────────────┘   get_obs()       └──────────────────┘   state/image     └────────┘
  它自己不做控制细节，控制全部委托给 MotionController；自己也不懂相机协议，
  图像要么来自 RealSense 直连（legacy），要么来自 ROS bridge（bridge）。

观测来源有两个，务必分清：
  * legacy：相机用 RealSense 直连（MultiCameraManager），关节用机器人 WebSocket。
            图像时间戳和关节时间戳由本文件自己"就近对齐"。
  * bridge：图像+关节都从 ROS 的 bridge WebSocket 订阅，由 bridge 侧已做好的
            TopicAligner 对齐成一对观测，本文件只管取用。

控制链路（自底向上）大致是：
    transport.WebsocketTransport → 单帧收发
    interpolation.LinearInterpolator → 两帧之间平滑过渡
    motion.MotionController ← env 直接对话的总指挥（含后台 300Hz 发布线程）
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from PIL import Image

from tron2_env.config import Tron2Config
from tron2_env.joints import JointIndex
from tron2_env.motion import MotionController, create_motion_controller
from tron2_env.bridge import BridgeConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CameraConfig:
    """相机配置（仅 legacy / RealSense 直连模式用到）

    dataclass 小科普：Python 的 dataclass 会自动生成 __init__，字段带默认值。
    所以 `CameraConfig()` 就能得到一个全默认配置对象，也可以按需覆盖某个字段：
        CameraConfig(resolution=(720, 1280, 3), save_debug_images=False)
    """
    # 相机名称 (统一使用 obs 输出名称)
    # 这里的名字必须与 MultiCameraManager 的 serial_to_name 映射后的名称一致，
    # 也决定了 get_obs() 返回的 images 字典的 key。
    camera_names: List[str] = field(default_factory=lambda: [
        "cam_high",           # 头顶俯视相机（D455 等）
        "cam_left_wrist",     # 左手腕相机（D405 等）
        "cam_right_wrist",    # 右手腕相机（D405 等）
    ])

    # 相机分辨率 (H, W, C) —— 高 480、宽 640、RGB 三通道
    resolution: Tuple[int, int, int] = (480, 640, 3)

    # 最大队列大小
    # 每个相机在独立线程里持续拉帧并塞进一个 deque(maxlen=...)，
    # 消费者总是取"最新一帧"。队列满时旧帧自动被挤出，保证数据新鲜。
    max_queue_size: int = 10

    # 是否保存调试图像
    save_debug_images: bool = True
    debug_image_dir: str = "./debug_images"




@dataclass
class EnvConfig:
    """环境配置 —— 把所有可调项集中在一个 dataclass 里

    新手使用姿势：
        config = EnvConfig(
            robot_config=Tron2Config(robot_ip="192.168.1.10"),
            observation_source="legacy",   # 或 "bridge"
            publish_rate=300.0,            # 机器人控制发布频率
        )
        env = Tron2Env(config)
    """
    # 机器人配置（IP、端口、上电位姿 init_joints、安全参数等，见 config.py）
    robot_config: Tron2Config = field(default_factory=Tron2Config)

    # 相机配置（legacy 模式用）
    camera_config: CameraConfig = field(default_factory=CameraConfig)

    # 控制后端: 当前公开版本只支持 "websocket"
    # 保留这个参数是为了将来可以换别的传输实现（见 motion/__init__.py 的说明）。
    control_backend: str = "websocket"

    # MotionController 后台 publish 频率 (Hz)。两个后端推荐 300Hz。
    # 含义：后台线程每秒向机器人发 300 次关节指令。频率越高指令越平滑，
    # 但也更吃带宽。机器人侧 servoj 是 PD 位置环，重复发同一个目标不会抖动。
    publish_rate: float = 300.0

    # consumer 节拍 / 命令到达目标的预期耗时 = 1/fps,MotionController 的
    # LinearInterpolator 用这个 ETA 在两次 command_joints 之间平滑过渡。
    # 换句话说：策略每 1/30 秒下发一个动作（30Hz），控制器保证在这段时间内
    # 从上一目标线性插值到新目标，指令之间不会出现跳变。
    fps: float = 30.0

    # 时间同步容差 (秒)
    # legacy 模式下，图像时间戳和关节时间戳差多少以内就认为"同一时刻"，
    # 不必再去找更接近的帧。超出的部分交由 _sync_observation 处理。
    time_sync_tolerance: float = 0.01
    time_sync_max_retries: int = 3
    legacy_use_time_sync: bool = True

    # 夹爪初始化开口度 (0-1)
    # reset() 时把夹爪张开到这个比例（1 表示全开）。夹爪指令在底层是 0-100。
    init_gripper_opening: float = 0.9

    # 原始配置字典（用于透传给其他组件）
    # 例如 MultiCameraManager.from_config(raw_config) 会从这里的 "camera" 键读取参数。
    raw_config: Dict[str, Any] = field(default_factory=dict)

    # 观测来源: "legacy" (RealSense 直连) | "bridge" (WebSocket bridge)
    #   legacy：图像走 RealSense，关节走机器人直连 WebSocket（见 get_obs 注释）
    #   bridge：图像+关节都从 ROS bridge 订阅，控制仍走机器人直连
    observation_source: str = "legacy"

    # 状态维度: 16 (双臂+夹爪) 或 18 (双臂+夹爪+头部)
    # 注意这里的"16/18"是环境对外暴露 state 的最大维度。
    # 与 joints.py 里的两个常量对应：SERVOJ_DIM=16（伺服指令）和 STATE_DIM=18（完整状态）。
    # 头部如果不参与控制，就截断为 16。
    state_dim: int = 16

    # Bridge 模式下 state 来源: "bridge" 使用 bridge 对齐 state，"legacy" 使用机器人直连 state
    #   bridge_state_source="bridge"：state 直接用 bridge 对齐好的 18 维状态
    #   bridge_state_source="legacy"：图像仍走 bridge，但 state 从机器人直连 WebSocket 现拉
    #                                 此时会关闭 bridge 的 joint/gripper 订阅，只用它的图像
    bridge_state_source: str = "bridge"

    # Bridge WebSocket 配置（observation_source="bridge" 时生效）
    # 见 bridge.py 的 BridgeConfig：host / 话题名 / 对齐延迟上限 / 是否存图等。
    bridge_config: BridgeConfig = field(default_factory=BridgeConfig)


# ============================================================================
# Tron2 Environment
# ============================================================================

class Tron2Env:
    """Tron2机器人环境

    这是整个包对外的主要入口，负责：
      1. 创建并启动机器人控制器（MotionController）
      2. 创建观测提供者（bridge 或 legacy 二选一）
      3. 提供标准的 reset / step / get_obs 接口给上层策略使用

    Examples:
        >>> config = EnvConfig(robot_config=Tron2Config(robot_ip="ROBOT_IP"))
        >>> env = Tron2Env(config)
        >>> obs = env.reset()
        >>> action = np.zeros(16)  # 16维动作
        >>> env.step(action)
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        """初始化环境

        Args:
            config: 环境配置，如果为None则使用默认配置
        """
        self.config = config or EnvConfig()

        # 设置日志
        self._setup_logger()

        # 初始化机器人(MotionController = transport + interpolator + publish loop)
        # create_motion_controller 是工厂函数，一步到位返回一个"已经启动好"的控制器：
        #   建 WebsocketTransport → 组装 MotionController → start()（读当前关节角作
        #   插值起点、启动后台 300Hz 发布线程）。见 motion/controller.py。
        self.logger.info(
            "正在初始化机器人控制器 (backend=%s, publish_rate=%.0fHz, fps=%.1f)...",
            self.config.control_backend,
            self.config.publish_rate,
            self.config.fps,
        )
        self.robot: MotionController = create_motion_controller(
            self.config.robot_config,
            backend=self.config.control_backend,
            publish_rate=self.config.publish_rate,
            # eta = 走完一个动作的预期时长 = 1/fps ≈ 33ms (30Hz)。
            # 传给插值器，让它在两次 command_joints 之间线性过渡到新目标。
            eta_default=1.0 / max(self.config.fps, 1e-6),
        )

        # 初始化观测来源：bridge 与 legacy 二选一，绝不会同时开启。
        if self.config.observation_source == "bridge":
            self.logger.info("观测来源: Bridge WebSocket")
            # 校验 bridge_state_source 的取值，防止拼写错误悄悄走错分支。
            if self.config.bridge_state_source not in {"bridge", "legacy"}:
                raise ValueError(
                    f"bridge_state_source must be 'bridge' or 'legacy', got {self.config.bridge_state_source!r}"
                )
            if self.config.bridge_state_source == "legacy":
                self.logger.info("Bridge 模式 state 来源: Legacy robot WebSocket")
                # 既然 state 走机器人直连，就不再需要 bridge 帮忙订阅 joint/gripper 话题，
                # 只保留图像订阅（省掉一份无用流量和带宽）。
                if self.config.bridge_config.joint_topics:
                    self.logger.info("Bridge legacy-state 模式: 禁用 bridge joint/gripper 订阅，仅使用 bridge 图像")
                    self.config.bridge_config.joint_topics = {}
            self.camera_manager = None                # bridge 模式不需要 RealSense
            self.bridge_provider = self._init_bridge()
        else:
            self.logger.info("观测来源: Legacy (RealSense 直连)")
            self.camera_manager = self._init_camera()  # 启动三台 RealSense 采集线程
            self.bridge_provider = None                # legacy 模式不需要 bridge

        # 状态管理
        self.last_action: Optional[np.ndarray] = None  # 最近一次下发的 16 维伺服目标
        self.init_joints = self.config.robot_config.init_joints  # 上电初始位姿(14维手臂)

        # 创建调试图像目录（按当前观测来源选择对应的配置）
        # Path.mkdir(parents=True, exist_ok=True) 表示递归创建、已存在也不报错。
        if self.config.observation_source == "bridge":
            if self.config.bridge_config.save_debug_images:
                Path(self.config.bridge_config.debug_image_dir).mkdir(parents=True, exist_ok=True)
        else:
            if self.config.camera_config.save_debug_images:
                Path(self.config.camera_config.debug_image_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("环境初始化完成")

    def _setup_logger(self):
        """设置日志系统

        给本类配一个独立的 logger（名字叫 "Tron2Env"），并且：
          - 只有没有 handler 时才添加（避免重复注册产生重复输出）；
          - propagate=False：不让日志继续向根 logger 冒泡（防止与别的组件重复打印）。
        """
        self.logger = logging.getLogger("Tron2Env")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def _init_bridge(self):
        """初始化 Bridge WebSocket 观测提供者

        BridgeObservationProvider（见 bridge.py）做的是：
          后台线程跑 asyncio 事件循环，同时订阅多路图像话题 + 关节话题，
          用 TopicAligner 按"最近图像时间戳"对齐，产出 {images, state, metadata}，
          塞进线程安全队列；主线程 get_obs() 从队列取最新一帧。
        """
        try:
            from tron2_env.bridge import BridgeObservationProvider
        except ImportError:
            # 没有装 websockets 等依赖时给出明确提示，而不是报一个莫名其妙的名字错误。
            self.logger.error("无法导入 BridgeObservationProvider，请确保 websockets 已安装")
            raise

        provider = BridgeObservationProvider(self.config.bridge_config)
        provider.start()  # 启动后台订阅线程（daemon 线程）

        # 等待首次观测到达：5 秒内能取到一帧说明链路通；取不到也只警告不阻塞，
        # 因为 bridge 可能在重连中，稍后自然会有数据。
        self.logger.info("等待 Bridge 观测就绪...")
        try:
            provider.get_obs(timeout=5.0)
        except TimeoutError:
            self.logger.warning("Bridge 首次观测超时，将继续运行")
        self.logger.info("Bridge 观测就绪")

        return provider

    def _init_camera(self):
        """初始化相机管理器（legacy 模式）

        MultiCameraManager（见 camera.py）：为每台 RealSense 相机开一个独立线程，
        持续 wait_for_frames 并把最新帧压入 deque；主线程随时可取"最新帧"，
        不会被慢相机的 wait_for_frames 阻塞。
        """
        try:
            from tron2_env.camera import MultiCameraManager
        except ImportError:
            self.logger.error("无法导入 MultiCameraManager，请确保 pyrealsense2 已安装")
            raise

        # 尝试从 YAML 加载（如果存在配置字典）
        # 若用户通过 raw_config 传了 camera 配置（序列号映射、分辨率等），
        # 就优先用 from_config 构造；否则用 camera_config 里的默认参数。
        if hasattr(self.config, 'raw_config'):
            camera_manager = MultiCameraManager.from_config(self.config.raw_config)
        else:
            camera_manager = MultiCameraManager(
                max_queue_size=self.config.camera_config.max_queue_size
            )

        camera_manager.start_capture()  # 识别串口 → 配置 pipeline → 逐个开线程

        # 等待相机预热
        # 刚启动的相机前几帧可能黑屏或参数未就绪，睡 2 秒让采集线程先跑起来。
        self.logger.info("相机预热中...")
        time.sleep(2.0)

        return camera_manager

    # ========================================================================
    # Environment Interface
    # ========================================================================

    def reset(self) -> Dict:
        """重置环境到初始状态

        三步走：
          1) 读一帧观测，顺便核对图像分辨率、机器人是否在初始位置；
          2) 若偏离初始位置超过容差，调用 wait_until_reached 让机器人回位；
          3) 下发一帧"只动夹爪、手臂保持当前"的动作，把夹爪张到 init_gripper_opening。

        Returns:
            初始观测
        """
        self.logger.info("重置环境...")

        # 获取当前观测
        obs = self.get_obs()

        # 验证图像尺寸（仅 legacy 模式）
        # bridge 模式的图像来自 ROS，尺寸由 bridge 侧保证，不需要这里校验。
        if self.config.observation_source != "bridge":
            expected_shape = self.config.camera_config.resolution
            for cam_name in self.config.camera_config.camera_names:
                actual_shape = obs['images'][cam_name].shape
                if actual_shape != expected_shape:
                    self.logger.warning(
                        f"{cam_name} 分辨率不匹配: 期望{expected_shape}, 实际{actual_shape}"
                    )

        # 验证机器人位置：若不在 init_joints，就阻塞等待它走回去（回位）
        if self.init_joints is not None:
            current_state = obs['state']
            # 把左右两段手臂(7+7=14维)拼起来，与 init_joints(14维)逐元素比较。
            # JointIndex.LEFT_ARM / RIGHT_ARM 是 slice(0,7) / slice(8,15)，见 joints.py。
            arm_states = np.concatenate([
                current_state[JointIndex.LEFT_ARM],
                current_state[JointIndex.RIGHT_ARM]
            ])
            init_arm = np.array(self.init_joints)

            # 取逐元素差值的最大值：只要有一个关节偏了 5 厘米/度以上就算"未就位"。
            error = np.abs(arm_states - init_arm).max()
            if error > 0.05:
                self.logger.warning(f"机器人未在初始位置，最大误差: {error:.4f}")
                # 阻塞等待 movej 回位。wait_until_reached 内部会持续发整段轨迹，
                # 直到每个关节误差 < tolerance 或超时。
                self.robot.wait_until_reached(self.init_joints,tolerance=0.05)

        # 初始化夹爪
        # 技巧：复制当前 state 当作"动作"，只改夹爪两维、其余保持现状，
        # 这样手臂不会因为复位动作而乱动。夹爪开度从 state 直接替换为期望值。
        test_action = obs['state'].copy()
        test_action[JointIndex.LEFT_GRIPPER] = self.config.init_gripper_opening
        test_action[JointIndex.RIGHT_GRIPPER] = self.config.init_gripper_opening
        self.step(test_action)  # 非阻塞，夹爪指令立即发出

        self.logger.info("环境重置完成")
        return obs

    def step(self, action: Union[List[float], np.ndarray]):
        """执行一个动作.

        非阻塞:把 16-dim 目标更新到 MotionController 的内部 interpolator,
        publish 线程会在 ~1/fps 时间窗内平滑过渡到该目标。夹爪走单独通路。

        新手理解要点：
          - 这里"执行动作"不等于"等待动作完成"。它只是把目标写进插值器，
            真正的逐帧发送由后台 300Hz 线程完成，所以调用几乎是瞬间返回的。
          - 策略循环通常是：obs = env.get_obs() → action = policy(obs) →
            env.step(action) → 等 1/fps 秒 → 下一轮。动作频率 fps 决定了 ETA。

        Args:
            action: 16/18维动作向量
                   - 16维: [7关节+1夹爪(左), 7关节+1夹爪(右)]
                   - 18维: [7关节+1夹爪(左), 7关节+1夹爪(右), 2头部]
                   （各段的具体下标见 joints.py 的 JointIndex）
        """
        # 输入验证
        # 先把 list 转成 np.ndarray，统一后续按切片取段的写法。
        if isinstance(action, list):
            action = np.array(action)

        # 维度必须合法：SERVOJ_DIM=16 或 STATE_DIM=18，其它一律拒绝。
        if len(action) not in [JointIndex.SERVOJ_DIM, JointIndex.STATE_DIM]:
            raise ValueError(f"动作维度应为{JointIndex.SERVOJ_DIM}/{JointIndex.STATE_DIM}, 实际{len(action)}")

        # 提取双臂关节动作 (14-dim)
        # 把左臂 7 维和右臂 7 维按 [L_arm(7), R_arm(7)] 的顺序拼起来，
        # 对应 servoj 指令（16 维）的前 14 维。
        arm_action = np.concatenate([
            action[JointIndex.LEFT_ARM],
            action[JointIndex.RIGHT_ARM]
        ])

        # 提取头部动作 (2-dim: pitch, yaw)
        if len(action) >= JointIndex.STATE_DIM:
            # 18 维动作：头部在最后两维，直接取。
            head_action = action[JointIndex.HEAD]
        else:
            # 16 维动作：不含头部，沿用当前头部位置 (transport 缓存,无锁泄漏)
            # get_head_position 读的是 transport 侧缓存的最新头部值，不加锁也不阻塞。
            head_action = self.robot.get_head_position()

        # 组合为 16 维 servoj 设定点 (14 臂 + 2 头)
        # 注意：servoj 指令里【没有夹爪】！夹爪是单独一条通路（set_gripper）。
        # 这个 16 维向量与 obs['state'] 的 16 维截断版布局一致。
        full_servo_action = np.concatenate([arm_action, head_action])

        # 夹爪 (归一化 0..1 → 0..100)
        # 动作里的夹爪是 0~1 开度，transport 的 set_gripper 期望 0~100 开度。
        # 乘 100 再 clip 到 [0, 100]，防止非法值（如 NaN 或越界）发到机器人。
        gripper_action = np.clip(
            np.array([action[JointIndex.LEFT_GRIPPER], action[JointIndex.RIGHT_GRIPPER]]) * 100.0,
            0, 100,
        )
        # 夹爪走专用通路：不经过插值器，立即发送（夹爪是开度式控制，不需要平滑轨迹）。
        self.robot.set_gripper(
            left_opening=gripper_action[0],
            right_opening=gripper_action[1],
        )

        # 更新 publish loop 的目标。MotionController 用 eta=1/fps 在两次
        # command_joints 之间线性插值,所以这里不需要 env 自己再做插值。
        # command_joints 只是 set 插值器的 destination，立即返回（非阻塞）。
        self.robot.command_joints(full_servo_action)
        self.last_action = full_servo_action  # 记录，方便上层调试/回放

    def get_obs(self) -> Dict:
        """获取当前观测

        Returns:
            观测字典: {
                'state': np.ndarray,  # 关节状态 (16/18维)
                'images': Dict[str, np.ndarray]  # 图像字典
            }

        metadata 中关于时间戳的约定：
        - ``joint_timestamp_ms`` / ``gripper_timestamp_ms`` 始终对应 ``state``
          字段实际的来源（"我们推理用的那帧 state 的时间戳"）。
        - ``state_source`` 取值 ``bridge`` 或 ``legacy``，指示 state 来自哪条路径。
        - 若 ``state_source == "legacy"`` 但图像走 bridge，``bridge_joint_timestamp_ms``
          / ``bridge_gripper_timestamp_ms`` 会保留 bridge 自己对齐到的关节时间戳，
          供调试对比，不参与正常推理。

        新手要点：这个方法的两个分支本质是"两条完全不同的取数路径"——
          bridge 模式：obs 是 bridge 后台线程对齐好的一整包（含 state+images）；
          legacy 模式：图像与关节各来自不同设备，需要按时间戳对齐合成一包。
        """
        # Bridge 模式：图像来自 bridge，可选 state 来自 bridge 或机器人直连
        if self.config.observation_source == "bridge":
            # 从 bridge 线程安全队列取最新对齐好的观测（1 秒超时）。
            obs = self.bridge_provider.get_obs(timeout=1.0)
            metadata = dict(obs.get("metadata", {}))

            if self.config.bridge_state_source == "legacy":
                # state 走 robot ws 直拉。bridge 已经在 TopicAligner 里对齐过
                # 图像-state,这里直接信任 robot ws 自己的 timestamp,不再做二次
                # 对齐 —— 之前的 _sync_observation 调用会把 bridge ts 与 robot ws
                # ts 强行拼到一起,反而引入不同源时钟的失配,并多出 ~5ms sleep。
                #
                # 简单说：如果 state 是刚现拉的，它的时间戳就以 robot ws 为准，
                # bridge 给的关节时间戳只保留在 metadata 里供对比。
                bridge_joint_timestamp_ms = metadata.get("joint_timestamp_ms")
                bridge_gripper_timestamp_ms = metadata.get("gripper_timestamp_ms")
                # 从机器人直连 WebSocket 现拉 18 维状态（最多等 0.5 秒）。
                qpos_dict = self.robot.get_joint_states(timeout=0.5)
                # 只取前 state_dim 维（16），截掉不用的头部维度。
                obs["state"] = np.asarray(
                    qpos_dict["states"][:self.config.state_dim], dtype=np.float32
                )
                metadata.update({
                    "state_source": "legacy",
                    "bridge_joint_timestamp_ms": bridge_joint_timestamp_ms,
                    "bridge_gripper_timestamp_ms": bridge_gripper_timestamp_ms,
                    "joint_timestamp_ms": qpos_dict.get("timestamp"),
                    "gripper_timestamp_ms": qpos_dict.get("timestamp"),
                })
            else:
                # 直接用 bridge 对齐好的 state，截断到 state_dim 维。
                obs["state"] = obs["state"][:self.config.state_dim]
                metadata["state_source"] = "bridge"
            obs["metadata"] = metadata

            # 可选的调试存图（每次 get_obs 都存三张 JPG，便于肉眼确认画面）。
            if self.config.bridge_config.save_debug_images:
                self._save_debug_images_bridge(obs)
            return obs

        # Legacy 模式：先拿图像，用三相机中最旧的时间戳作为 obs 参考时刻，
        # 再在 200Hz joint_state_queue 里找与该时刻最近的 joint 帧。
        # 时间基准统一为客户端 time.time()（camera 和 transport 都用同一时钟）。
        #
        # 为什么用"最旧"的图像时间戳？因为观测要"同时"才有意义——参考时刻取
        # 三帧里最早的那帧，能保证所有图像都 ≥ 这个时刻；再拿这个时刻去关节
        # 队列里找"最近的关节帧"，得到的就是时间上最接近的一组观测。
        obs_start = time.time()
        # 1. 获取图像（每个相机最新的那一帧，附带 time.time() 时间戳）
        rgb_images = self._get_images()

        # 保存调试图像
        if self.config.camera_config.save_debug_images:
            self._save_debug_images(rgb_images)

        # 2. 确定 obs 参考时间戳 = 三相机中最旧的那帧（保证所有图像都 ≥ 该时刻）
        cam_timestamps = []
        for cam_name in self.config.camera_config.camera_names:
            ts_key = f'{cam_name}_timestamp'
            if ts_key in rgb_images:
                cam_timestamps.append(rgb_images[ts_key])
        if not cam_timestamps:
            # 一台相机都没有帧 = 采集线程没起来 / 相机没接好，属于致命错误。
            raise RuntimeError("No camera frames available in legacy mode")
        img_timestamp = min(cam_timestamps)  # 最旧的那帧

        # 3. 获取关节状态——在 joint_state_queue 里找与 img_timestamp 最近的帧
        # 机器人的 WebSocket transport 在后台以 200Hz 把最新关节帧压进队列
        # （maxlen=7，约 35ms 历史窗口）。find_nearest_state 在窗口内做
        # "最近时间戳"检索，返回离 img_timestamp 最近的一帧（不移除队列）。
        synced_qpos: Optional[Dict] = None
        if self.config.legacy_use_time_sync:
            synced_qpos = self.robot.find_nearest_state(img_timestamp)
            if synced_qpos is not None:
                joint_timestamp = synced_qpos['timestamp'] / 1000.0
                self.logger.debug(
                    "legacy obs sync: nearest queued joint_img=%.1fms",
                    (joint_timestamp - img_timestamp) * 1000.0,
                )
        # Fallback: 队列空 / 不启用 sync —— 走原始 popleft 取最新帧
        # 取队列里最新的一帧（从头部弹出），作为兜底方案。
        qpos_dict = synced_qpos
        if qpos_dict is None:
            qpos_dict = self.robot.get_joint_states(timeout=0.5)
            synced_qpos = qpos_dict
        joint_timestamp = qpos_dict['timestamp'] / 1000.0

        self.logger.debug(
            "legacy obs: ref_img=%.3fs joint=%.3fs diff=%.1fms",
            img_timestamp, joint_timestamp,
            (joint_timestamp - img_timestamp) * 1000.0,
        )

        # 4. 构建观测
        # 只保留配置里声明过的相机，避免意外字段混进 images。
        images = {}
        for cam_name in self.config.camera_config.camera_names:
            if cam_name in rgb_images:
                images[cam_name] = rgb_images[cam_name]
        # 每个相机各自的时间戳（毫秒），供上层做更精细的时间分析。
        image_timestamps_ms = {
            cam_name: int(rgb_images[f'{cam_name}_timestamp'] * 1000)
            for cam_name in self.config.camera_config.camera_names
            if f'{cam_name}_timestamp' in rgb_images
        }
        # obs 的"参考图像时间" = 最旧那帧的时间（毫秒）。
        image_timestamp_ms = (
            min(image_timestamps_ms.values()) if image_timestamps_ms else int(img_timestamp * 1000)
        )
        joint_timestamp_ms = synced_qpos.get('timestamp')
        synced_joint_timestamp = (joint_timestamp_ms or 0) / 1000.0
        obs_end = time.time()
        # 三台相机时间戳的跨度：跨度大说明相机间未对齐 / 有帧延迟。
        image_span_ms = 0.0
        if image_timestamps_ms:
            image_span_ms = max(image_timestamps_ms.values()) - min(image_timestamps_ms.values())

        self.logger.debug(
            "legacy obs timing: sync=%s raw_joint_img=%.1fms synced_joint_img=%.1fms "
            "img_age=%.1fms joint_age=%.1fms total=%.1fms image_span=%.1fms",
            self.config.legacy_use_time_sync,
            (joint_timestamp - img_timestamp) * 1000.0,
            (synced_joint_timestamp - img_timestamp) * 1000.0,
            (obs_end - img_timestamp) * 1000.0,
            (obs_end - synced_joint_timestamp) * 1000.0 if synced_joint_timestamp > 0 else float("nan"),
            (obs_end - obs_start) * 1000.0,
            image_span_ms,
        )

        obs = {
            "state": np.array(synced_qpos['states'][:self.config.state_dim]),
            "images": images,
            "metadata": {
                "state_source": "legacy",
                "observation_ref_timestamp_ms": image_timestamp_ms,
                "bridge_ref_timestamp_ms": image_timestamp_ms,
                "joint_timestamp_ms": joint_timestamp_ms,
                "gripper_timestamp_ms": joint_timestamp_ms,
                "image_timestamp_ms": image_timestamp_ms,
                "image_timestamps_ms": image_timestamps_ms,
                "legacy_initial_joint_timestamp_ms": qpos_dict.get('timestamp'),
                "legacy_time_sync_enabled": self.config.legacy_use_time_sync,
            },
        }

        return obs

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _get_images(self) -> Dict:
        """获取相机图像

        从 MultiCameraManager 取每台相机"最新的一帧"，并把 BGR 转成 RGB
        （RealSense 默认输出 BGR，而模型/可视化通常约定 RGB）。

        Returns:
            图像字典: {
                'cam_high': np.ndarray,
                'cam_high_timestamp': float,
                ...
            }
        """
        all_frames = self.camera_manager.get_all_latest_frames()
        image_dict = {}

        for camera_name, frame_data in all_frames.items():
            if frame_data is not None:
                # BGR转RGB：numpy 切片 [:, :, ::-1] 把通道轴倒过来 (BGR→RGB)。
                image_dict[camera_name] = frame_data['color'][:, :, ::-1]
                # 时间戳是采集线程记下的 time.time()（与 transport 同一时钟）。
                image_dict[f'{camera_name}_timestamp'] = frame_data['timestamp']

        return image_dict

    def _sync_observation(
        self,
        img_timestamp: float,
        initial_qpos: Dict,
        using_sync: bool = False
    ) -> Dict:
        """同步观测时间戳

        策略: 直接在 transport 的 200Hz joint_state_queue (maxlen=7, ~35ms 历史窗口) 中
        查询与 img_timestamp 时间最近的关节帧——非阻塞，不消耗队列，避免重试 sleep。

        Args:
            img_timestamp: 图像时间戳 (秒)
            initial_qpos: get_obs 已 popleft 的最新关节状态 (作为 fallback)

        Returns:
            同步后的关节状态

        新手理解：这是"硬对齐"的兜底逻辑。get_obs 里的主线流程其实已经
        调用了 find_nearest_state，这个函数是历史版本留下的双保险——
        如果当前帧差太多，就到历史队列里翻一翻有没有更接近图像时刻的帧。
        """
        joint_timestamp = initial_qpos['timestamp'] / 1000.0
        time_dif = joint_timestamp - img_timestamp

        if not using_sync:
            self.logger.debug("legacy obs sync disabled: joint_img=%.1fms", time_dif * 1000.0)
            return initial_qpos

        # 已经在容差内，直接用
        if abs(time_dif) <= self.config.time_sync_tolerance:
            self.logger.debug("legacy obs sync ok: joint_img=%.1fms", time_dif * 1000.0)
            return initial_qpos

        # 在队列里找与 img_timestamp 最近的帧
        nearest = self.robot.find_nearest_state(img_timestamp)
        if nearest is None:
            self.logger.debug(
                "legacy obs sync: no queued state available; using initial (joint_img=%.1fms)",
                time_dif * 1000.0,
            )
            return initial_qpos

        nearest_ts = nearest['timestamp'] / 1000.0
        nearest_dif = nearest_ts - img_timestamp
        # 只有当队列里的帧比 initial_qpos 更接近 image_ts 才换用
        # 否则维持 initial（换到更差的那帧反而引入更多误差）。
        if abs(nearest_dif) < abs(time_dif):
            self.logger.debug(
                "legacy obs sync: switched to nearer queued state (initial=%.1fms, nearest=%.1fms)",
                time_dif * 1000.0,
                nearest_dif * 1000.0,
            )
            return nearest

        self.logger.debug(
            "legacy obs sync: initial already nearest (initial=%.1fms, queued_best=%.1fms)",
            time_dif * 1000.0,
            nearest_dif * 1000.0,
        )
        return initial_qpos

    def _save_debug_images(self, rgb_images: Dict):
        """保存调试图像（legacy 模式）

        把三台相机的当前帧分别存成 ./debug_images/<相机名>.jpg，
        方便肉眼确认画面内容、方向、遮挡等，不用专门开一个 viewer。
        """
        debug_dir = Path(self.config.camera_config.debug_image_dir)

        for key in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
            if key in rgb_images:
                img = Image.fromarray(rgb_images[key])  # 需要 RGB，这里已是 RGB
                save_path = debug_dir / f"{key}.jpg"
                img.save(save_path)

    def _save_debug_images_bridge(self, obs: Dict):
        """保存调试图像（bridge 模式）

        与 _save_debug_images 类似，只是输入是 bridge 给的整包 obs，
        遍历其中的 images 字典逐个保存。
        """
        debug_dir = Path(self.config.bridge_config.debug_image_dir)

        for cam_name, image in obs.get("images", {}).items():
            img = Image.fromarray(image)
            save_path = debug_dir / f"{cam_name}.jpg"
            img.save(save_path)

    def close(self):
        """关闭环境并释放资源

        清理顺序很重要：
          1. 机器人：先停发布线程再断开 transport（优雅停机，见 MotionController.disconnect）；
          2. 相机：停止各采集线程，pipeline.stop() 释放 USB 带宽；
          3. bridge：发停止信号并 join 后台线程。
        只有 close() 后，脚本退出才不会留下"僵尸线程/占用相机"的残留。
        """
        self.logger.info("关闭环境...")

        if hasattr(self, 'robot'):
            self.robot.disconnect()

        if hasattr(self, 'camera_manager') and self.camera_manager is not None:
            self.camera_manager.stop_capture()

        if hasattr(self, 'bridge_provider') and self.bridge_provider is not None:
            self.bridge_provider.stop()

        self.logger.info("环境已关闭")

    def __enter__(self):
        """上下文管理器入口

        支持 `with Tron2Env() as env:` 写法；退出 with 块时自动调用 close()。
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()


# ============================================================================
# Policy Wrapper (Example)
# ============================================================================
# 下面两个类是"示例"性质的策略包装器：它们把 env 产出的 obs 喂给一个远端
# 策略服务（WebSocket 客户端），拿回动作给 env.step。这不在 env 的核心链路里，
# 只是给用户演示"观测 → 策略 → 动作 → env"完整闭环的写法模板。

class PolicyWrapper:
    """策略包装器基类

    定义策略的统一接口：给定 obs，返回动作。子类实现真正的推理逻辑
    （本地模型 / 远端服务 / 手写规则都行）。
    """

    def get_action(self, observation: Dict) -> np.ndarray:
        """获取动作

        Args:
            observation: 观测字典

        Returns:
            动作数组
        """
        raise NotImplementedError  # 基类只声明接口，必须由子类实现


class WebsocketPolicyWrapper(PolicyWrapper):
    """基于WebSocket的策略客户端

    连到一个已部署的策略服务（openpi 的 websocket 服务端），把观测发过去，
    远端推理完把一串动作返回。
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        """初始化WebSocket策略客户端

        Args:
            host: 服务器地址
            port: 服务器端口
        """
        try:
            # openpi_client 是 OpenPI 项目提供的策略客户端库。
            from openpi_client import websocket_client_policy, image_tools
            self.ws_client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
            self.image_tools = image_tools  # 存起来，推理前预处理图像用
        except ImportError as e:
            # 依赖缺失时给出可操作的错误提示。
            raise ImportError(f"无法导入 openpi_client: {e}")

        self.logger = logging.getLogger("WebsocketPolicy")

    def get_action(self, observation: Dict) -> np.ndarray:
        """通过WebSocket获取动作

        Args:
            observation: 观测字典

        Returns:
            动作序列 (action_horizon, action_dim)
        """
        import einops

        # 预处理图像
        # 策略服务通常要求固定尺寸 (224x224) 且为 [C, H, W] 排布：
        #   resize_with_pad：等比缩放+补边到 224x224；
        #   convert_to_uint8：转成 uint8（保留 0-255）；
        #   einops.rearrange "h w c -> c h w"：把 HWC 转成 CHW（PyTorch 惯例）。
        obs = observation.copy()
        for cam_name in obs["images"]:
            img = self.image_tools.convert_to_uint8(
                self.image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        # 推理
        result = self.ws_client.infer(obs)
        actions = np.stack(result['actions'], axis=0)

        return actions
