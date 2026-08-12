"""Tron2 dual-arm robot control + environment library.

本包是 Tron2 双臂机器人"控制 + 强化学习环境"的库入口。
对新手最重要的第一件事：**这个文件本身不实现任何逻辑**，它只是一个
"总目录 / 门面（facade）"——把所有对外可用的类集中在这里重新导出，
让用户写代码时只需要 `import tron2_env`，不必关心类各自住在哪个子模块。

建议按**自底向上**的顺序阅读整个库（下层的被上层依赖）：

  * ``transport/``     — 最底层：与机器人的"单发单收"原语（WebSocket JSON）
                        只负责收发一帧数据，不做限频、不做插值。
  * ``interpolation/`` — 纯数学：关节轨迹插值（把目标关节角平滑过渡过去）。
  * ``motion/``        — 承上启下：MotionController = transport + 插值器
                        + 后台发布线程（按 publish_rate 高频发指令）。
  * ``env.py``         — 最上层：Tron2Env（gym 风格环境，封装 MotionController，
                        对外暴露 step()/reset()，供强化学习 / 策略推理使用）。
  * ``bridge.py``      — 可选：从 ROS bridge WebSocket 订阅"图像 + 关节"观测。
  * ``camera.py``      — 可选：RealSense 多相机管理（每相机独立采集线程）。
  * ``rtc/``           — 可选：Real-Time Chunking 辅助工具（ActionQueue 动作队列、
                        LatencyTracker 延迟统计），用于低延迟动作流式执行。
"""

# ---------------------------------------------------------------------------
# 集中的"重新导出"区
# ---------------------------------------------------------------------------
# 注意：下面这些 import 只是把子模块里定义好的名字"引过来再暴露出去"。
# 这样做的三个好处：
#   1. 用户只需 `import tron2_env` 就能拿到全部常用类；
#   2. 公共 API 一目了然（不用翻整个包去找）；
#   3. 将来重构内部文件结构时，只要保持这里的导出不变，用户的代码就不受影响。
# ---------------------------------------------------------------------------

from tron2_env.bridge import BridgeConfig          # bridge 观测的配置项（ws 地址 / 订阅话题等）
from tron2_env.config import Tron2Config           # 机器人连接与初始化参数（robot_ip / port / 初始位姿）
from tron2_env.env import (                        # 环境层：gym 风格封装 + 配套配置
    CameraConfig,            # 相机配置（相机名 / 分辨率 / 队列大小）
    EnvConfig,               # 环境总配置（机器人配置 + 相机配置等）
    PolicyWrapper,           # 策略包装器基类（定义 get_action 接口的"约定"）
    Tron2Env,                # 核心环境类：step()/reset()，控制委托给 MotionController
    WebsocketPolicyWrapper,  # 基于 WebSocket 的策略客户端（通过 openpi_client 获取动作）
)
from tron2_env.errors import (                     # 统一异常体系（详见 errors.py）
    CommandError,            # 指令（servoj / 夹爪等）校验或发送失败
    ConnectionError,         # 无法建立或维持机器人连接
    StateError,              # 状态读取失败（超时 / 维度错误等）
    Tron2Error,              # 所有运行时异常的基类，用于统一捕获
)
from tron2_env.interpolation import JointInterpolator, LinearInterpolator  # 插值器接口 + 线性插值实现
from tron2_env.joints import JointIndex            # 关节索引常量（维度布局、切片位置），全库共用
from tron2_env.motion import MotionController, create_motion_controller   # 控制器类 + 工厂函数
from tron2_env.transport import RobotTransport, WebsocketTransport        # 传输接口 + WebSocket 实现

# ---------------------------------------------------------------------------
# __all__：声明"公开 API"清单
# ---------------------------------------------------------------------------
# 作用：
#   1. 控制 `from tron2_env import *` 会导入哪些名字（只导入列出来的）；
#   2. 相当于给 IDE / 文档工具一份"这是官方公开接口"的清单。
# 新手可以把它当成**该包的 API 索引**：想找某个能力，先来这查名字。
# 注意：其中 `BridgeObservationProvider` 和 `MultiCameraManager` 并没有在上方
# 真正 import —— 它们由文件底部的 __getattr__ 延迟提供（见下方说明）。
# ---------------------------------------------------------------------------
__all__ = [
    # config / joints / errors —— 配置、常量、异常
    "Tron2Config",
    "JointIndex",
    "Tron2Error",
    "ConnectionError",
    "CommandError",
    "StateError",
    # transports —— 传输层（接口 + WebSocket 实现）
    "RobotTransport",
    "WebsocketTransport",
    # interpolation —— 插值层
    "JointInterpolator",
    "LinearInterpolator",
    # motion —— 控制层
    "MotionController",
    "create_motion_controller",
    # env —— 环境层
    "Tron2Env",
    "EnvConfig",
    "CameraConfig",
    "PolicyWrapper",
    "WebsocketPolicyWrapper",
    # bridge + camera (lazy) —— 可选依赖，延迟加载
    "BridgeConfig",
    "BridgeObservationProvider",
    "MultiCameraManager",
]


def __getattr__(name):
    """Lazy import for optional deps (RealSense / websockets).

    这是 Python 3.7 引入的"模块级属性缺失兜底钩子"（PEP 562）：
    当上面的普通 import 里找不到 ``name`` 时，Python 会自动调用本函数来"补救"。

    为什么要这样做？
    - MultiCameraManager 依赖 pyrealsense2（重量级、可能没装）；
      BridgeObservationProvider 依赖 websocket/异步环境（同样可选）。
    - 如果直接在顶部 import，那么即使你只用 MotionController，
      也必须先装齐这些可选依赖，否则整个包都 import 失败。
    - 改成延迟加载后：**只有当你真的访问这两个名字时**才去 import 对应模块。
      平时 `import tron2_env` 又轻又快，也不会因缺依赖而报错。

    使用方式：`from tron2_env import MultiCameraManager` 和直接
    `import tron2_env; tron2_env.MultiCameraManager` 都会触发这里的逻辑。

    若名字也不在此列，说明真的不存在，抛出标准的 AttributeError。
    """
    if name == "MultiCameraManager":
        from tron2_env.camera import MultiCameraManager

        return MultiCameraManager
    if name == "BridgeObservationProvider":
        from tron2_env.bridge import BridgeObservationProvider

        return BridgeObservationProvider
    raise AttributeError(f"module 'tron2_env' has no attribute {name}")
