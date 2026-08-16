"""Latency tracking utilities for Real-Time Chunking (RTC).

================================================================================
给新手的快速导读：这个文件在做什么？
================================================================================

1. 背景：为什么需要"延迟统计"？
   ----------------------------------------------------------------------
   RTC 的 delay（推理期间过去了多少帧）直接决定动作块怎么对齐，而推理耗时
   本身是**波动**的：JAX 首次调用要做 JIT 重编译（可能一下子卡好几秒），
   之后才进入稳定状态。想要"统计出一个合理的 delay 值"或"监控推理是否健康"，
   就必须持续测量推理耗时，并回答两个问题：
     * 最坏情况多长？（决定超时 / 缓冲区怎么设）   → 用 max
     * 常态下多长？（决定 delay 用多少，比如取 p95）→ 用分位数

   本类就是做这件事的：推理线程每完成一次推理就 add() 一个耗时样本，
   之后随时可以查询 max / p95 / 均值等统计量。

2. 两个关键设计（理解了它就读懂了这个类）
   ----------------------------------------------------------------------
   a) 滑动窗口（deque + maxlen）
      我们只关心"最近一段时间"的延迟：久远的样本对当下的决策毫无意义。
      collections.deque(maxlen=N) 塞满后会自动丢掉最旧的样本，天然就是一个
      滑动窗口，且两端操作都是 O(1)，比 list 随手实现更省心。

   b) "分位数裁剪（cap）"与"max 不裁剪"的区分
      一次冷启动 / JIT 重编译会让某个样本飙到好几秒。如果直接算 p95，
      这个孤立尖峰会把稳态值拉高，导致 delay 估计偏保守。
      → 计算分位数时，先把每个样本"夹紧"到 max_cap_s（如 0.6 秒），
        让尖峰不至于污染稳态统计（percentile 内部做 np.minimum）。
      → 但 max() 不受影响：最坏情况必须反映真实值（不夹紧），因为它是
        用来设计超时 / 缓冲的，不能"自欺欺人"。

3. 与 ActionQueue 的关系（怎么用）
   ----------------------------------------------------------------------
   ActionQueue.merge 需要一个 real_delay 来裁掉"已过去的帧"。
   real_delay 的一个常见取值来源就是这里的 p95：
     推理有 95% 的概率不会超过 p95，用它做 delay 能覆盖绝大多数情况，
     只在极端尖峰（>p95）时略微滞后。
   两个类不互相引用：LatencyTracker 由推理循环调用，ActionQueue 由控制线程
   调用，二者在时序上由上层（如 Tron2Env / 推理循环）协调。

4. 线程安全提醒
   ----------------------------------------------------------------------
   本类内部**没有加锁**（与 ActionQueue 不同）。典型用法是推理线程调用
   add()，由同一线程或监控线程查询统计值；若确定会被多个线程同时读写，
   需要调用方在外部自行加锁（单次 append / 单次迭代在 CPython 的 GIL 下
   是原子的，但 add() 与 max()/percentile() 并发时可能读到瞬时不一致——
   对监控统计而言通常可以接受）。
================================================================================
"""

import logging
from collections import deque

import numpy as np

# 模块级日志器：按模块名分层，便于用
# "tron2_env.rtc.latency_tracker" 单独控制本模块的日志级别。
logger = logging.getLogger(__name__)


class LatencyTracker:
    """Tracks recent latencies and provides max/percentile queries.

    记录最近若干次推理延迟，提供 max / 分位数（p50 / p95 等）查询。

    两种"读口径"对比：
      * max()       —— 历史最坏情况（不裁剪，反映真实峰值，用于超时/缓冲设计）；
      * percentile()—— 稳态 / 典型值（先裁剪到 max_cap_s 再统计，抗冷启动尖峰，
                        用于给 real_delay 取一个稳健的估计）。

    Args:
        maxlen: 滑动窗口大小。给出后只保留最近 ``maxlen`` 个样本，
            自动淘汰最旧的；为 None 则保留全部历史（一般不需要）。
        max_cap_s: 每个样本在参与分位数统计前的上限（单位：秒）。用于压制
            JAX JIT 重编译等冷启动尖峰。原始样本仍被完整记录，只有分位数
            查询会对每个样本裁剪到该上限。设为 None 或 <=0 表示禁用裁剪。
    """

    def __init__(self, maxlen: int = 100, max_cap_s: float | None = 0.6):
        # _values：滑动窗口容器。deque(maxlen=N) 塞满后自动丢最旧样本，
        #           因此"窗口内样本数"永远 <= maxlen。
        # _max_cap_s：裁剪上限；传入 <=0 视为"未启用"，归一化为 None
        #             （这样后续判断只用检查 is not None 一种情况）。
        self._values = deque(maxlen=maxlen)
        self._max_cap_s = max_cap_s if (max_cap_s is None or max_cap_s > 0) else None
        self.reset()

    def reset(self) -> None:
        """清空所有已记录的延迟样本（新任务开始时调用）。"""
        self._values.clear()
        # 运行期维护的"全局最大值"也一并清零。
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        """记录一个延迟样本（单位：秒）。"""
        val = float(latency)
        # 防御：负的"延迟"没有物理意义（可能是测量误差 / 时钟回拨），直接丢弃。
        if val < 0:
            return
        self._values.append(val)
        # 增量维护运行最大值：每次都全量遍历代价高，这里只与旧最大值取较大者。
        self.max_latency = max(self.max_latency, val)
        logger.debug("LatencyTracker: added %.1fms, max=%.1fms, count=%d",
                     val * 1000, self.max_latency * 1000, len(self._values))

    def __len__(self) -> int:
        # 支持 len(tracker) 语法，返回当前窗口内的样本个数。
        return len(self._values)

    def max(self) -> float | None:
        """返回历史最大延迟。

        注意两点：
          1. 返回的是"运行期维护的全局最大值"，**不会**因窗口滚动而衰减——
             即使那个最大样本已经被滑出窗口，这里仍会返回它；
             若只想看"当前窗口内"的最大值，需要自行遍历 _values。
          2. 从未 add 过任何样本时，返回初始值 0.0（而非 None，
             与签名中 "float | None" 的类型提示略有出入）。
        """
        return self.max_latency

    def percentile(self, q: float) -> float | None:
        """返回已记录延迟的 q 分位数（q 取值 [0,1]），例如 0.95 = 95 分位。

        处理流程：
          1. 样本为空 → 返回 0.0。这是**有意为之**：让下游可以直接拿结果做
             算术（如减法）而不报错；注意与签名中 "None" 的提示略有出入；
          2. 若启用了裁剪，先把每个样本夹紧到 max_cap_s——这样一次冷启动 /
             JIT 重编译尖峰不会把稳态估计拉高，而原始 max_latency 不受影响；
          3. q <= 0 返回（裁剪后的）最小值，q >= 1 返回（裁剪后的）最大值，
             其余用 numpy 的线性插值分位数。
        """
        if not self._values:
            return 0.0
        q = float(q)
        # float32：毫秒级延迟精度完全够用，且内存 / 计算更省。
        vals = np.array(list(self._values), dtype=np.float32)
        # 关键点：先裁剪、后统计，尖峰只影响"被夹住的那个值"本身。
        if self._max_cap_s is not None:
            vals = np.minimum(vals, self._max_cap_s)
        if q <= 0.0:
            return float(vals.min())
        if q >= 1.0:
            return float(vals.max())
        return float(np.quantile(vals, q))

    def p95(self) -> float | None:
        """返回 95 分位延迟（percentile(0.95) 的便捷封装）。

        p95 的含义：95% 的样本 <= 这个值。用它做 RTC 的 delay 时，
        有 95% 的推理能对齐上，只剩 5% 的极端尖峰需要额外容忍。
        """
        return self.percentile(0.95)

    def summary(self) -> str:
        """返回一行人类可读的统计摘要，便于直接打印 / 打日志。

        字段说明（单位均为毫秒）：
          n       —— 样本数；
          mean    —— 均值（未裁剪）；
          p50     —— 中位数（未裁剪）；
          p95     —— 95 分位（未裁剪）；
          p95_capped —— 额外附注：裁剪到 max_cap_s 后的 95 分位
                        （仅当启用了裁剪且窗口非空时出现），
                        用于对比"剔除尖峰后的稳态表现"；
          max     —— 历史最坏情况（未裁剪）。
        """
        if not self._values:
            return "LatencyTracker: no data"
        vals = np.array(list(self._values), dtype=np.float32)
        capped_note = ""
        if self._max_cap_s is not None:
            # 单独算一份"裁剪后"的 p95，作为对比用的附注。
            capped = np.minimum(vals, self._max_cap_s)
            capped_note = (
                f", p95_capped({self._max_cap_s*1000:.0f}ms)={np.quantile(capped, 0.95)*1000:.1f}ms"
            )
        return (
            f"LatencyTracker: n={len(vals)}, "
            f"mean={vals.mean()*1000:.1f}ms, "
            f"p50={np.quantile(vals, 0.5)*1000:.1f}ms, "
            f"p95={np.quantile(vals, 0.95)*1000:.1f}ms{capped_note}, "
            f"max={self.max_latency*1000:.1f}ms"
        )
