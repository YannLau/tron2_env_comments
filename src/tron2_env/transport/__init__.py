"""Robot transport layer — single-shot communication primitives.

这是 **transport（传输层）包的入口文件**——它本身几乎不含逻辑，
作用是把本子包内定义的两个重要名字"集中导出"，方便外部用一行
`from tron2_env.transport import ...` 直接拿到，而不用关心内部文件结构。

为什么要单独设一个子包？分层思想：
    transport/
        __init__.py   ← 本文件：包门面（只做导出）
        base.py       ← 定义"契约"：RobotTransport（Protocol 接口，见其注释）
        websocket.py  ← 给出"实现"：WebsocketTransport（真正连机器人的类）

新手建议先读 base.py 理解"接口长什么样"，再读 websocket.py 看"具体怎么实现"，
本文件只需知道"两个名字从哪来、导出了什么"即可。

The public runtime currently exposes the WebSocket JSON protocol used by the
TRON2 controller.
当前公开运行环境只暴露 TRON2 控制器使用的 WebSocket JSON 协议。
"""

# ---------------------------------------------------------------------------
# 重新导出（re-export）
# ---------------------------------------------------------------------------
# 下面的 import 只是把别的模块里已定义好的对象"引过来再暴露出去"，
# 两个名字分别代表这个子包的两面：
#   * RobotTransport      —— 接口（Protocol）：定义"一个 transport 应该会哪些方法"。
#                            你写代码时应该**依赖它**（类型注解、isinstance 检查）。
#   * WebsocketTransport  —— 实现：真正通过 ws://<ip>:5000 连机器人的类。
#                            你实际干活时用它（创建实例、调用方法）。
# 把两者放在同一个包里，就是告诉使用者："接口和默认实现是一对，配套使用。"
# ---------------------------------------------------------------------------
from tron2_env.transport.base import RobotTransport
from tron2_env.transport.websocket import WebsocketTransport

# __all__：声明本包对外公开的 API 白名单。
# 作用：
#   1. 约束 `from tron2_env.transport import *` 会导入哪些名字；
#   2. 作为"官方接口清单"给 IDE / 文档 / 使用者看。
# 注意 __init__.py 里能用的"隐藏文件"（如 base、websocket 模块本身）不在其中，
# 它们仍可显式 import，只是不进入通配导入。
__all__ = [
    "RobotTransport",
    "WebsocketTransport",
]
