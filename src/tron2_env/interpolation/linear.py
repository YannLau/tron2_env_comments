"""Linear interpolation between joint waypoints.

线性插值实现。这是 interpolation/base.py 里 ``JointInterpolator`` 协议的
**具体实现**（接口 vs 实现的关系，见 base.py 注释）。

**本文件在整条链路里的角色：**
它就是一个"会随时间在关节角之间走直线"的小对象。MotionController 的发布
线程每个节拍问它 current()，它就按"当前时刻 → 起点终点连线的位置"返回一个
16 维关节角数组。因此：

    时间 t = t_start    → 返回 q_start（起点）
    时间 t = t_end      → 返回 q_end（终点）
    时间 t 在中间       → 按线性比例返回 q_start + alpha * (q_end - q_start)

**为什么叫"一阶/线性"？** 因为关节角随时间按一条直线过渡（速度恒定），
没有加速度概念。对很短的 eta（约一个控制周期 1/fps，如 1/30 s）来说，
曲线根本来不及"弯"，线性就够用了；更高阶的曲线（三次样条等）在这么短的
时间内反而发挥不出平滑的优势。本文件只做线性。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

# 判断"两个时刻相等"的极小阈值。因为 t_start 和 t_end 相减时浮点误差可能
# 不是严格的 0，用 _EPS 作容差：只要差比它小就视为"同一时刻/已到达"。
_EPS = 1e-9


class LinearInterpolator:
    """First-order linear interpolation between consecutive waypoints.

    Suitable for short eta (~1/fps) where higher-order curves wouldn't have
    room to differentiate. Behaviour mirrors what the env used to do with
    ``np.linspace + servoj`` but at the publish_rate granularity instead of
    a fixed ``trajectory_points`` count.

    适合短 eta（~1/fps）场景：越短的时间越不需要高阶曲线。行为等价于旧环境
    里用 ``np.linspace + servoj`` 做的事，但粒度从"固定轨迹点数"改成了
    "按 publish_rate 逐拍取点"。
    """

    def __init__(self) -> None:
        # ---- 线程安全的关键 ----
        # 发布线程（后台）会并发调用 current() 来读，env.step（主线程）会并发
        # 调用 set_destination() 来写。用同一把锁保护所有内部状态，保证
        # "读时不会看到写了一半的数据"。这也兑现了 base.py 接口里
        # "实现必须线程安全"的约定。
        self._lock = threading.Lock()

        # ---- 描述"一条线段"的四个量 ----
        # 一段插值本质上就是一条从 (t_start, q_start) 到 (t_end, q_end) 的线段：
        #   q_start / q_end : 起点 / 终点的关节角（16 维向量）
        #   t_start / t_end : 起点 / 终点的时刻（秒，基于 time.perf_counter）
        # 只要知道这四个量，就能算出任意时刻 t 对应的关节角（见 _current_unlocked）。
        self._t_start = 0.0
        self._t_end = 0.0
        self._q_start: Optional[np.ndarray] = None
        self._q_end: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ public

    def reset(self, q: np.ndarray) -> None:
        # 播种：把"起点"和"终点"都设成 q，t_start = t_end = now。
        # 结果：此后 current() 永远返回 q（因为 span 为 0 → 走"已到达"分支），
        # 直到第一次 set_destination 才真正开始移动。
        # np.asarray(...).copy() 确保内部持有一份**独立副本**——调用方之后再
        # 改传入数组，不会污染我们内部的轨迹状态。
        q_arr = np.asarray(q, dtype=np.float64).copy()
        now = time.perf_counter()
        with self._lock:
            self._t_start = now
            self._t_end = now
            self._q_start = q_arr
            self._q_end = q_arr.copy()

    def set_destination(
        self,
        target: np.ndarray,
        eta: float,
        *,
        now: Optional[float] = None,
    ) -> None:
        # 重新定向：把目的地改成 target，用 eta 秒走完。
        target_arr = np.asarray(target, dtype=np.float64).copy()
        if now is None:
            now = time.perf_counter()
        with self._lock:
            if self._q_end is None:
                # 边界情况：还没 reset 就先 set_destination（例如直接用
                # set_destination 完成首次初始化）。此时把起点终点都设为 target，
                # 相当于"从 target 出发去 target"——立即到达，不动。
                self._q_start = target_arr.copy()
                self._q_end = target_arr.copy()
                self._t_start = now
                self._t_end = now
                return
            if target_arr.shape != self._q_end.shape:
                raise ValueError(
                    f"target shape {target_arr.shape} != current {self._q_end.shape}"
                )
            # ---- 兑现 base.py 注释里承诺的"无折痕（kink-free）" ----
            # 新起点取 current(now)：也就是**上一段轨迹在"此刻"已经走到的实际
            # 值**（_current_unlocked(now)，见下），而不是"上一个终点"。于是新线段
            # 从旧线段当前的连续位置出发，切换命令的瞬间不会发生关节角突变。
            self._q_start = self._current_unlocked(now).copy()
            self._q_end = target_arr
            self._t_start = now
            # max(eta, 0.0)：防御负 eta（把它当成 0，即立即到达）。
            self._t_end = now + max(eta, 0.0)

    def current(self, t: Optional[float] = None) -> np.ndarray:
        # 公开入口：加锁后转给无锁实现。t 缺省取"现在"。
        with self._lock:
            return self._current_unlocked(t)

    def at_destination(self, t: Optional[float] = None) -> bool:
        # 是否已到达：t >= t_end。注意"到达"指时刻已过终点时刻，
        # 而不是"误差足够小"——这是按时间表判定的，不是按位置判定的。
        if t is None:
            t = time.perf_counter()
        with self._lock:
            return t >= self._t_end

    # ----------------------------------------------------------------- private

    def _current_unlocked(self, t: Optional[float]) -> np.ndarray:
        # 核心数学：算任意时刻 t 对应的关节角。调用方必须已持有 _lock
        # （约定：方法名带 _unlocked 后缀 = 不自己加锁，由调用者负责加锁）。
        if self._q_end is None or self._q_start is None:
            # 还没初始化（reset 或 set_destination 都没调用过）就询问位置 → 报错。
            raise RuntimeError("interpolator not initialised — call reset() or set_destination()")
        if t is None:
            t = time.perf_counter()
        span = self._t_end - self._t_start   # 本次过渡的总时长 = eta
        if span <= _EPS or t >= self._t_end:
            # 分支 1：已经到时间了（或时长近 0）→ 直接返回终点 q_end。
            return self._q_end.copy()
        if t <= self._t_start:
            # 分支 2：还没开始（早于起点时刻）→ 返回起点 q_start。
            # 防御时钟倒退 / 调用方传了个过去的时间。
            return self._q_start.copy()
        # 分支 3：正常区间。alpha ∈ [0, 1] 表示"已走过的进度比例"，
        # 关节角 = 起点 + 比例 * (终点 - 起点)，即两点间的线性插值。
        alpha = (t - self._t_start) / span
        return self._q_start + alpha * (self._q_end - self._q_start)
