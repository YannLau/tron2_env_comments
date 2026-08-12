"""JointInterpolator — pure math primitive between waypoints.

这是插值子包里的"契约定义"文件，与 transport/base.py 的作用完全一致：
只声明**接口长什么样**，不提供任何实现——真正的线性实现放在 linear.py。

**先回答新手最核心的问题：这个接口到底在描述什么？**
它描述一个"**会随时间把关节角从起点平滑过渡到终点**的对象"。用一句话
概括整个交互模型：

    1. reset(q)           —— 把起点和终点都设成 q（相当于"对表"，告诉它我从哪出发）
    2. set_destination(target, eta) —— 改目的地：eta 秒内从"现在的位置"走到 target
    3. current(t)         —— 问它"在 t 时刻，关节应该到哪了？"（发布线程每 tick 问一次）
    4. at_destination(t)  —— 问它"t 时刻到没到目的地？"

Owned by :class:`tron2_env.motion.MotionController`. The controller asks
``current(t)`` once per publish tick to drive the transport; callers update
the destination via ``set_destination(target, eta)``.
这个对象归 MotionController 所有：控制器每个发布节拍调用一次 current(t)，
把得到的关节值发给传输层；调用方（环境）通过 set_destination(target, eta)
来更新目的地。

Implementations must be thread-safe (publish loop reads, env.step writes).
**实现必须线程安全**：因为发布线程（后台，读 current）和 env.step
（主线程，写 set_destination）是并发访问同一个对象的，见 linear.py
里 LinearInterpolator 用一把 Lock 保护全部内部状态的做法。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


# 和 RobotTransport 一样，这里用 Protocol 定义接口，而不是抽象基类：
#   不要求实现类显式声明继承关系，只要方法签名齐全就算"是"插值器。
# 这样 LinearInterpolator（乃至你以后自己写的三次样条插值器）都不必
# 依赖本文件，MotionController 照样能识别并驱动它们。
# 详见 transport/base.py 里对 Protocol / 结构子类型的完整讲解。
@runtime_checkable
class JointInterpolator(Protocol):
    """Smoothly interpolate joint targets between waypoints over time.

    在两个关节目标之间随时间平滑过渡的接口契约。
    """

    def reset(self, q: np.ndarray) -> None:
        """Seed both start and destination to ``q`` — ``current()`` will return ``q`` until first set_destination.

        把起点和终点都初始化为 q（"播种"）。
        调用之后、第一次 set_destination 之前，current() 会一直返回 q——
        也就是说：机器人会"保持现状不动"，等待你下达第一个目标。
        典型用途：MotionController.start() 在启动发布线程之前，先把当前
        实际关节角 reset 进去，避免发布循环一启动就把机器人"拽"到错误位置。
        """

    def set_destination(
        self,
        target: np.ndarray,
        eta: float,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Pre-emptively retarget.

        重新定向（非阻塞）：只改目的地，不等待到达。

        语义：
            q_start = current(now)   # 起点 = "现在"轨迹上的实际位置
            q_end   = target         # 终点 = 新目标
            t_start = now
            t_end   = now + eta      # 花 eta 秒从起点平滑走到终点

        Calling this in the middle of an in-flight interpolation produces a
        kink-free curve because q_start is captured from the current
        trajectory value, not the last destination.

        **为什么能做到"无折痕"（kink-free）？**
        想象上一条轨迹还没走完、你就下达了新目标。如果起点直接取"上一个
        终点"，关节角会在命令切换瞬间发生突变（折痕）。而这里的做法是
        起点取 current(now)——也就是**上一段轨迹在"此刻"已经走到的实际值**——
        新的插值从那个连续的位置继续，曲线没有突变。
        这正是强化学习 / 实时控制中"每条控制周期都要重新定向"场景的关键。
        """

    def current(self, t: Optional[float] = None) -> np.ndarray:
        """Return q at wall-time ``t`` (perf_counter; default = now).

        返回在墙上时间 t 时刻的关节角。t 缺省时取"现在"。
        注意时间基准是 time.perf_counter()（单调时钟，适合测时长），
        不是 time.time()（墙上时钟，可能被校时跳变）。
        """

    def at_destination(self, t: Optional[float] = None) -> bool:
        """True once t >= t_end (motion has reached its target).

        判断"是否已到达目的地"：当 t >= t_end（即过了设定的到达时刻）
        返回 True。可用于知道"插值是否已完成"（例如确定可以安全读取
        最终关节角）。
        """
