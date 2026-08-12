"""Motion control orchestration — owns transport + interpolator + publish loop.

这是 **motion（运动控制）子包的入口文件**。和前两个子包的 `__init__.py`
一样，它是个"包门面"：本身几乎不含逻辑，只负责把 `controller.py` 里定义
的两个名字集中导出。

**这个子包在整条控制链路里扮演什么角色？**
它是链路的最顶层——**环境（env）真正直接对话的对象**。回顾完整链路（自底向上）：

    RobotTransport（传输）     → 只负责收发单帧
    JointInterpolator（插值）  → 在两个关节目标之间"平滑过渡"
    MotionController（控制）   ← 本子包：持有 transport + interpolator + 发布线程

前面两个子包都是"接口 + 实现"的成对结构（base.py 定义契约、websocket.py /
linear.py 给出实现），而本子包**没有 base.py**。原因：MotionController 是
消费方（它内部依赖 transport 接口和 interpolator 接口），本身是控制栈的
顶点、由环境直接 new 出来用，不需要再抽象成"可以被替换的接口"。

**MotionController 到底在干什么？** 一句话概括：
启动时读一次真实关节角作为插值起点，然后开一个后台发布线程，按 publish_rate
(默认 300Hz) 每个节拍问插值器 `current(t)`，把结果发给 transport 推给机器人；
外部调用 `command_joints(target)` 只更新插值器的目的地（非阻塞），线程会自动
把机器人平滑带到目标。

子包内部：
    motion/
        __init__.py   ← 本文件：包门面（只做导出）
        controller.py  ← 全部逻辑：MotionController 类 + create_motion_controller 工厂
"""

# ---------------------------------------------------------------------------
# 重新导出（re-export）
# ---------------------------------------------------------------------------
# 从 controller.py 引进来再暴露出去的，是两个配合使用的名字：
#   * MotionController           —— 控制器的"本体"：一个具体类，持有 transport +
#                                   插值器 + 发布线程。环境拿它来发指令、读状态。
#   * create_motion_controller   —— 工厂函数：传一个 Tron2Config（机器人 IP、
#                                   端口、初始化位姿等），它会创建 WebsocketTransport、
#                                   组装 MotionController、调 start() 启动发布线程，
#                                   一步到位返回一个**已就绪**的控制器。
# 所以两者的分工是：
#   想"从头一步一步自己搭" → 用类；
#   想"一行代码拿到能用的控制器" → 用工厂。
# 工厂是新手入门最快的入口（examples/mock_quickstart.py 里就是这么用的，
# 只不过那里传入的是一个内存假 transport，不是真 websocket）。
# ---------------------------------------------------------------------------
from tron2_env.motion.controller import MotionController, create_motion_controller

# __all__：本子包的公开 API 白名单。与前两个子包同理：
#   1. 约束 `from tron2_env.motion import *` 的导入范围；
#   2. 作为"官方接口清单"给 IDE / 文档 / 使用者参考。
# controller 模块文件本身不在此列，但仍可显式 import
# （如 `from tron2_env.motion.controller import MotionController` 同样可行）。
__all__ = ["MotionController", "create_motion_controller"]
