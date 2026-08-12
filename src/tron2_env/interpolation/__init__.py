"""Joint interpolation primitives — used by ``MotionController``.

这是 **interpolation（插值）子包的入口文件**，和 transport/__init__.py 一样
是个"包门面"：本身几乎不含逻辑，只负责把子包内定义的两个名字集中导出。

**这个子包在整条控制链路里扮演什么角色？**
回顾一下完整链路（自底向上）：

    RobotTransport（传输）   → 只负责收发单帧
    JointInterpolator（插值） ← 本子包：在两个关节目标之间"平滑过渡"
    MotionController（控制）  → 每个发布节拍问一次插值器 current(t)，拿去发给机器人

为什么要插值？因为如果直接把 16 维目标一次性丢给机器人，关节会瞬间跳过去、
产生猛冲甚至危险动作。插值器的作用就是**把"当前关节角"平滑地过渡到"目标
关节角"**：发布线程每 tick 问它"现在应该到哪了？"，它按时间算出中间值，
机器人才会走出一条连续、可预测的轨迹。

子包内部：
    interpolation/
        __init__.py   ← 本文件：包门面（只做导出）
        base.py       ← 定义"契约"：JointInterpolator（Protocol 接口）
        linear.py     ← 给出"实现"：LinearInterpolator（线性插值）

与 transport 子包完全同构：base 管接口、linear 管实现、本文件管导出。
"""

# ---------------------------------------------------------------------------
# 重新导出（re-export）
# ---------------------------------------------------------------------------
# 和 transport 子包一样，两个名字分别是"接口"和"默认实现"：
#   * JointInterpolator —— 接口（Protocol）：描述"一个插值器应该会哪些方法"
#                           （reset / set_destination / current / at_destination）。
#   * LinearInterpolator —— 实现：第一阶（线性）插值，适合短 eta 场景。
# 上层 MotionController 默认持有一个 LinearInterpolator 实例，
# 但因为它依赖的是接口而不是具体类，将来换曲线插值（如三次样条）时，
# 只要新类满足同一接口，MotionController 完全不用改。
# ---------------------------------------------------------------------------
from tron2_env.interpolation.base import JointInterpolator
from tron2_env.interpolation.linear import LinearInterpolator

# __all__：本子包的公开 API 白名单。
# 与 transport/__init__.py 同理：
#   1. 约束 `from tron2_env.interpolation import *` 的导入范围；
#   2. 作为"官方接口清单"给 IDE / 文档 / 使用者参考。
# base / linear 两个模块文件本身不在此列，但仍可显式 import（如
# `from tron2_env.interpolation.linear import LinearInterpolator` 同样可行）。
__all__ = ["JointInterpolator", "LinearInterpolator"]
