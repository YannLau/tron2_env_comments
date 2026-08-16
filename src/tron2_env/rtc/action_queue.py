# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 LimX Dynamics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by LimX Dynamics in 2026 from Hugging Face LeRobot commit
# ca87ccd9413c59c30f524967222d2e3f1b7bb549. The modifications port queue
# storage from PyTorch tensors to NumPy arrays, replace RTCConfig coupling with
# runtime arguments, and add atomic snapshots, delay handling, and merge-boundary
# diagnostics for the TRON2 runtime.

"""Action queue management for Real-Time Chunking (RTC).

================================================================================
给新手的快速导读：这个文件在做什么？
================================================================================

1. 背景：为什么要"动作队列"？
   ----------------------------------------------------------------------
   策略模型（如 openpi / ACT）的输出方式有两种：
     * 单步输出：每次推理只给出"下一帧"的动作 → 每控制一个时间步都要等一次
       推理，端到端延迟高，控制频率受限；
     * 分块输出（RTC / Action Chunk）：一次推理给出"未来 T 帧"的动作块
       （shape = (T, action_dim)），机器人在接下来的 T 帧里逐帧取用，期间
       完全不用等推理 → 低延迟、高频控制。

   ActionQueue 就是用来存放这些"动作块"的线程安全队列：策略线程往里写
   （merge），控制线程往外逐帧取（get）。写和读发生在不同线程，因此内部
   用一把锁（Lock）保护。

2. 为什么有两份队列（queue 与 original_queue）？
   ----------------------------------------------------------------------
   RTC 的核心是"旧块 + 新块"的衔接：新块推理期间机器人还在执行旧块，所以
   新块要与旧块的进度对齐（见第 4 点的 delay）。对齐时需要比较新旧动作的
   "时间轴"，而这需要动作处于**模型坐标系（normalized 空间）**，因为拼接
   正确性依赖于归一化后的数值：
     * original_queue —— 模型原始输出（normalized 空间），仅用于 RTC 计算
       "上一块剩多少"（leftover），不直接发给机器人；
     * queue —— 后处理过、可直接下发给机器人的动作（真实物理空间，如关节
       弧度、夹爪指令），是控制线程真正消费的那份。
   两份队列"长度同步、内容互为映射"，merge 时同时更新。

3. 两种工作模式（rtc_enabled）
   ----------------------------------------------------------------------
     * RTC 模式（rtc_enabled=True） ：新块**整体替换**旧队列，并按推理延迟
       把"已经过去的帧"裁掉（_replace_actions_queue）。适合推理耗时波动大的
       场景，永远只执行最新规划。
     * 非 RTC 模式（rtc_enabled=False）：新块**追加**到队列尾部，维持连续
       动作流，不裁旧帧（_append_actions_queue）。适合推理稳定的场景。

4. delay 是什么意思？（RTC 论文里的 t - s）
   ----------------------------------------------------------------------
   时间线是这样的：
     s 时刻：策略看到观测 obs_s，开始推理新块 A_cur（耗时 d 帧）；
     s+d 时刻：推理完成，新块到达队列。而这 d 帧期间机器人一直在执行旧块，
              —— 所以 A_cur 的前 d 帧对应的"世界时间"已经过去了！
     因此把 A_cur 放进队列前，要裁掉前 d 帧，再从第 d 帧开始执行。
   这个 d 就是 delay。它有两种测量方式（详见 _check_and_resolve_delays）：
     * real_delay：推理代码按"墙钟"估算出的耗时（换算成帧数）；
     * 实际消费帧数：控制线程在推理期间真的取走了几帧
       （= last_index - action_index_before_inference）。
   后者的时钟与控制线程的读指针同源，更可信，因此本实现优先用它。

5. 建议的阅读顺序
   ----------------------------------------------------------------------
     先看 __init__ 理解两个成员队列和读指针 last_index；
     再看 get / qsize / empty（"读"端）；
     再看 merge（"写"端，核心）以及它调用的三个私有方法：
       _check_and_resolve_delays（怎么算 delay）
       _replace_actions_queue / _append_actions_queue（RTC / 非 RTC 怎么写）
     最后看 _compute_boundary_diagnostics / _delta_metrics（衔接质量诊断，
       纯日志用途，不影响控制逻辑，可最后理解）。

   本文件改编自 LeRobot 的 ActionQueue（原实现用 PyTorch 张量），本实现改用
   NumPy 数组。
================================================================================
"""

import logging
from threading import Lock

import numpy as np

# 模块级日志器：`logging.getLogger(__name__)` 会按模块名分层，
# 便于用户通过 "tron2_env.rtc.action_queue" 单独控制本模块的日志级别。
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TRON2 双机械臂的动作维度约定（仅用于下面的"衔接质量诊断"，不参与控制）
# ---------------------------------------------------------------------------
# TRON2 每个时间步的动作是 16 维：
#   0..6   左臂 7 个关节
#   7      左夹爪
#   8..14  右臂 7 个关节
#   15     右夹爪
# PROCESSED_ARM_INDICES 列出"双臂关节"的索引，PROCESSED_GRIPPER_INDICES
# 列出"夹爪"的索引。诊断时按这两组分别统计，方便区分"臂的跳变"和"夹爪的跳变"。
PROCESSED_ARM_INDICES = tuple(range(7)) + tuple(range(8, 15))
PROCESSED_GRIPPER_INDICES = (7, 15)


def _safe_dim(value: float, default: int = -1) -> int:
    """把可能是 NaN 的"维度号"转成可读整数，仅供日志格式化使用。

    诊断统计中某维出现 NaN 时（比如队列为空、拿不到对比样本），直接
    打印 NaN 会让日志难读；这里统一替换成 default（默认 -1），
    让日志输出形如 "max 0.1234@-1" 而不是 "@nan"。
    """
    if np.isnan(value):
        return default
    return int(value)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    线程安全的"动作块"队列，用于实时控制中管理策略输出的动作序列。

    两类动作序列（长度同步，内容互为映射）：
    - Original actions（original_queue）：模型原始输出，normalized 空间，
      仅用于 RTC 计算上一块的剩余（leftover）。
    - Processed actions（queue）：后处理后的动作，真实物理空间，
      控制线程真正消费并发给机器人。

    两种工作模式（由 rtc_enabled 决定）：
    1. RTC-enabled：新块整体替换队列，并按推理延迟裁掉已过去的帧；
    2. RTC-disabled：新块追加到队列尾部，保持动作连续性。

    线程模型：
    - 写线程（策略/推理线程）调用 merge() 写入新块；
    - 读线程（控制线程）调用 get() 逐帧取出；
    所有公共方法内部都用 self.lock 加锁，保证读写互斥、读读互斥。
    """

    def __init__(
        self,
        rtc_enabled: bool = False,
    ):
        # ------------------------------------------------------------------
        # 成员变量总览：
        #   queue           —— 后处理动作队列（(T, action_dim)，真实空间）
        #   original_queue  —— 原始动作队列（(T, action_dim)，normalized 空间）
        #   lock            —— 读写互斥锁（控制线程读 / 策略线程写）
        #   last_index      —— 读指针：下一个将要被 get() 取出的行下标
        #   rtc_enabled     —— 工作模式开关（替换 or 追加）
        #   _merge_count          —— 累计 merge 次数（仅用于日志/诊断）
        #   _last_merge_diagnostics —— 最近一次 merge 的衔接质量诊断
        # ------------------------------------------------------------------
        self.queue: np.ndarray | None = None  # Processed actions (T, action_dim)
        self.original_queue: np.ndarray | None = None  # Original actions for RTC (T, action_dim)
        self.lock = Lock()
        self.last_index = 0
        self.rtc_enabled = rtc_enabled
        self._merge_count = 0
        self._last_merge_diagnostics: dict[str, float] = {}

    def get(self) -> np.ndarray | None:
        """从队列中取出下一个动作（读端核心）。

        语义相当于"弹出 + 消费"：
          - 若队列为空，或读指针已越过队尾 → 返回 None（表示暂时没有可执行动作）；
          - 否则返回第 last_index 行，并把指针 +1，下次 get 取下一帧。
        返回的是 .copy()，防止调用方拿到内部数组的引用后原地修改，
        破坏队列其他位置的数据。

        （注：get 不清空已消费的行，队列"剩余长度"由 qsize() = len - last_index 表达。）
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index].copy()
            self.last_index += 1
            return action

    def clear(self) -> None:
        """清空队列并复位读指针（丢弃所有未执行的动作）。"""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """返回队列中"尚未被消费"的动作帧数。

        因为 get 不真正删除行，所以剩余量 = 总行数 - 已消费行数。
        """
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """判断队列是否已空（无剩余可消费动作）。"""
        if self.queue is None:
            return True
        return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """返回当前读指针 last_index（策略推理前常先取一次，用于计算 delay）。"""
        with self.lock:
            return self.last_index

    def snapshot_left_over(self) -> tuple[int, np.ndarray | None, int]:
        """原子地一次性快照 (读指针, 原始队列剩余动作, 队列剩余长度)。

        为什么必须"一次性"？RTC 计算 prev_chunk_left_over 时需要同时知道
        ``s``（读指针）和 ``A_cur[s:]``（该时刻原始队列剩余部分），二者必须
        来自**同一瞬间**。若分开调用 get_action_index() 再调 get_left_over()，
        两次调用之间读线程可能又取走了若干帧，导致拿到的快照自相矛盾
        （指针来自旧时刻、数据来自新时刻）。因此这里加锁、一起读、一起返回。

        返回值：
          - index：当时的读指针 s；
          - 原始队列 s 之后的部分（normalized 空间），队列为空则为 None；
          - qsize：当时剩余的已处理动作帧数。
        """
        with self.lock:
            index = self.last_index
            qsize = 0 if self.queue is None else len(self.queue) - index
            if self.original_queue is None:
                return index, None, qsize
            return index, self.original_queue[index:].copy(), qsize

    def get_left_over(self) -> np.ndarray | None:
        """返回"当前块未被消费的原始动作"（normalized 空间）。

        这些动作对应 RTC 论文里的 prev_chunk_left_over：上一块执行到
        last_index 后还没用完的部分。RTC 会把它与新块的衔接一起处理，
        用于修正两个块之间的连续性。
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].copy()

    def get_processed_left_over(self) -> np.ndarray | None:
        """返回"当前正在执行的已处理动作"的剩余部分（真实空间）。"""
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].copy()

    def get_last_merge_diagnostics(self) -> dict[str, float]:
        """返回最近一次 merge 的衔接质量诊断（只读副本，供日志/调参）。"""
        with self.lock:
            return dict(self._last_merge_diagnostics)

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: int | None = None,
        extra_delay: int = 0,
    ) -> int:
        """把策略新推理出的动作块写入队列（写端核心，被策略线程周期性调用）。

        Args:
            original_actions: 模型原始输出 (T, action_dim)，normalized 空间。
                不直接下发，仅存为 original_queue 供 RTC 计算 leftover。
            processed_actions: 后处理后的动作 (T, action_dim)，真实物理空间
                （关节弧度 / 夹爪指令等），存入 queue 供控制线程消费。
                与 original_actions 逐帧对应（同一时刻的两种表示）。
            real_delay: 本次推理的墙钟耗时折算成的帧数 d。表示新块的前 d 帧
                对应的时间点已经过去，应裁掉（见模块 docstring 第 4 点）。
            action_index_before_inference: 推理**开始前**一刻的读指针 s。
                用它算出的"推理期间实际消费的帧数"比 real_delay 更可靠
                （两个时钟同源）。为 None 时退回使用 real_delay。
            extra_delay: 额外裁掉的帧数，用于补偿"观测本身过期"（比如相机
                画面滞后于当前真实状态），通常由上层根据传感器延迟设置。

        Returns:
            最终采用的 delay（裁掉了多少帧），供上层记录/日志。
        """
        with self.lock:
            # ① 确定最终裁帧量：综合 real_delay / 实际消费帧数 / extra_delay，
            #    并做一致性校验（详见 _check_and_resolve_delays）。
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference, extra_delay)

            # ② 记录 merge 前的状态，并计算新旧块在衔接处的连续性诊断
            #    （仅供日志，不影响控制逻辑）。
            pre_qsize = len(self.queue) - self.last_index if self.queue is not None else 0
            pre_last_idx = self.last_index
            diagnostics = self._compute_boundary_diagnostics(original_actions, processed_actions, delay)

            # ③ 真正写入队列：RTC 模式整体替换 + 裁帧；否则追加 + 裁剪已消费前缀。
            if self.rtc_enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
            else:
                self._append_actions_queue(original_actions, processed_actions)

            # ④ 汇总诊断信息，供外部 get_last_merge_diagnostics() 读取。
            self._merge_count += 1
            diagnostics.update({
                "merge_count": float(self._merge_count),
                "merge_used_delay": float(delay),
                "merge_extra_delay": float(extra_delay),
                "merge_pre_qsize": float(pre_qsize),
                "merge_pre_index": float(pre_last_idx),
            })
            self._last_merge_diagnostics = diagnostics

            # ⑤ 调试日志：merge 前后队列规模、采用的 delay 等。
            post_qsize = len(self.queue) - self.last_index if self.queue is not None else 0
            logger.debug(
                "merge #%d: pre_qsize=%d, post_qsize=%d, pre_idx=%d, post_idx=%d, "
                "delay=%d, extra_delay=%d, rtc=%s, orig=%s, proc=%s",
                self._merge_count, pre_qsize, post_qsize, pre_last_idx, self.last_index,
                delay, extra_delay, self.rtc_enabled,
                original_actions.shape, processed_actions.shape,
            )
            # ⑥ 若诊断非空，打印衔接处的连续性指标（info 级，默认可见）。
            #    指标含义见 _delta_metrics 的返回值说明。
            if diagnostics:
                logger.info(
                    "boundary #%d: delay=%d extra=%d proc_plan=max %.4f@%d d=%.4f mae %.4f "
                    "proc_exec=max %.4f@%d d=%.4f mae %.4f "
                    "arm_plan=max %.4f@%d d=%.4f mae %.4f arm_exec=max %.4f@%d d=%.4f mae %.4f "
                    "grip_plan=max %.4f@%d d=%.4f mae %.4f raw_plan=max %.4f@%d d=%.4f mae %.4f",
                    self._merge_count,
                    delay,
                    extra_delay,
                    diagnostics.get("boundary_proc_plan_max", np.nan),
                    _safe_dim(diagnostics.get("boundary_proc_plan_max_dim", np.nan)),
                    diagnostics.get("boundary_proc_plan_max_delta", np.nan),
                    diagnostics.get("boundary_proc_plan_mae", np.nan),
                    diagnostics.get("boundary_proc_exec_max", np.nan),
                    _safe_dim(diagnostics.get("boundary_proc_exec_max_dim", np.nan)),
                    diagnostics.get("boundary_proc_exec_max_delta", np.nan),
                    diagnostics.get("boundary_proc_exec_mae", np.nan),
                    diagnostics.get("boundary_proc_plan_arm_max", np.nan),
                    _safe_dim(diagnostics.get("boundary_proc_plan_arm_max_dim", np.nan)),
                    diagnostics.get("boundary_proc_plan_arm_max_delta", np.nan),
                    diagnostics.get("boundary_proc_plan_arm_mae", np.nan),
                    diagnostics.get("boundary_proc_exec_arm_max", np.nan),
                    _safe_dim(diagnostics.get("boundary_proc_exec_arm_max_dim", np.nan)),
                    diagnostics.get("boundary_proc_exec_arm_max_delta", np.nan),
                    diagnostics.get("boundary_proc_exec_arm_mae", np.nan),
                    diagnostics.get("boundary_proc_plan_gripper_max", np.nan),
                    _safe_dim(diagnostics.get("boundary_proc_plan_gripper_max_dim", np.nan)),
                    diagnostics.get("boundary_proc_plan_gripper_max_delta", np.nan),
                    diagnostics.get("boundary_proc_plan_gripper_mae", np.nan),
                    diagnostics.get("boundary_raw_plan_max", np.nan),
                    _safe_dim(diagnostics.get("boundary_raw_plan_max_dim", np.nan)),
                    diagnostics.get("boundary_raw_plan_max_delta", np.nan),
                    diagnostics.get("boundary_raw_plan_mae", np.nan),
                )
            return delay

    def _compute_boundary_diagnostics(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        delay: int,
    ) -> dict[str, float]:
        """衡量"旧块 → 新块"在衔接处的连续性（诊断用，不影响控制）。

        衔接跳变是 RTC 的经典问题：新块到达时，机器人还在执行旧块的某个位置。
        如果旧块剩余的动作与新块开头的动作差异很大，机器人的轨迹就会出现
        "台阶式突变"。本函数把"旧块还剩的那帧"和"新块第一帧"做对比，量化跳变。

        对比口径（区分 plan 与 exec）：
          - old_next_proc：旧块中"下一个要执行"的帧（queue[last_index]）；
            它应和新块裁帧后的第一帧衔接 —— 对应指标后缀 *_plan_*。
          - old_prev_proc：旧块中"正在执行"的帧（queue[last_index-1]）；
            因为新块裁掉的 delay 帧正是这段"正在执行"的时间 —— 对应 *_exec_*。
          分别对"全部维度 / 手臂 / 夹爪"做统计，并用 raw（normalized 空间）
          也统计一份，便于多角度排查。

        Args:
            original_actions: 新块原始动作 (T, action_dim)。
            processed_actions: 新块后处理动作 (T, action_dim)。
            delay: 新块要裁掉的帧数。

        Returns:
            dict[str, float]：一组以 "boundary_*" 为前缀的指标，
            形如 {prefix}_mae / {prefix}_max / {prefix}_max_dim / {prefix}_max_delta。
        """
        diagnostics: dict[str, float] = {}
        # delay 可能超出新块长度（异常情况），先安全夹到合法区间，避免越界。
        clamped_delay = max(0, min(delay, len(original_actions), len(processed_actions)))

        # 新块裁帧后真正要执行的第一帧（processed 与 raw 各一份）
        new_proc = processed_actions[clamped_delay] if clamped_delay < len(processed_actions) else None
        new_raw = original_actions[clamped_delay] if clamped_delay < len(original_actions) else None
        # 旧块中"下一个要执行"的帧
        old_next_proc = (
            self.queue[self.last_index]
            if self.queue is not None and self.last_index < len(self.queue)
            else None
        )
        # 旧块中"正在执行"的帧（上一帧）
        old_prev_proc = (
            self.queue[self.last_index - 1]
            if self.queue is not None and self.last_index > 0 and self.last_index - 1 < len(self.queue)
            else None
        )
        # 旧块原始队列中"下一个要执行"的帧
        old_next_raw = (
            self.original_queue[self.last_index]
            if self.original_queue is not None and self.last_index < len(self.original_queue)
            else None
        )

        # 分别计算"新第一帧 vs 旧待执行帧 / 旧执行中帧"的差异指标，
        # 并针对手臂、夹爪单独再算一遍，便于定位突变发生在哪部分。
        diagnostics.update(self._delta_metrics("boundary_proc_plan", old_next_proc, new_proc))
        diagnostics.update(self._delta_metrics("boundary_proc_exec", old_prev_proc, new_proc))
        diagnostics.update(self._delta_metrics("boundary_raw_plan", old_next_raw, new_raw))
        diagnostics.update(self._delta_metrics("boundary_proc_plan_arm", old_next_proc, new_proc, PROCESSED_ARM_INDICES))
        diagnostics.update(self._delta_metrics("boundary_proc_exec_arm", old_prev_proc, new_proc, PROCESSED_ARM_INDICES))
        diagnostics.update(
            self._delta_metrics("boundary_proc_plan_gripper", old_next_proc, new_proc, PROCESSED_GRIPPER_INDICES)
        )
        diagnostics.update(
            self._delta_metrics("boundary_proc_exec_gripper", old_prev_proc, new_proc, PROCESSED_GRIPPER_INDICES)
        )
        # 记录新块裁帧后真正开始执行的位置（即 clamped_delay），便于日志对照。
        diagnostics["boundary_new_index"] = float(clamped_delay)
        return diagnostics

    @staticmethod
    def _delta_metrics(
        prefix: str,
        old: np.ndarray | None,
        new: np.ndarray | None,
        indices: tuple[int, ...] | None = None,
    ) -> dict[str, float]:
        """计算两帧动作之间的差异指标：MAE、最大差值及其位置。

        Args:
            prefix: 结果字典的键前缀（用于区分不同的对比口径）。
            old: 旧帧（如旧块待执行帧）。
            new: 新帧（如新块第一帧）。
            indices: 若给定，只在这几个维度上做比较（如只看手臂关节）；
                为 None 则比较全部维度。

        Returns:
            dict，键为 f"{prefix}_mae"（平均绝对误差）、
            f"{prefix}_max"（最大绝对差值）、
            f"{prefix}_max_dim"（最大差值所在维度号）、
            f"{prefix}_max_delta"（最大差值的带符号值）。
            任一侧为 None 或比较维度为 0 时，四个指标都置为 NaN（无法统计）。
        """
        if old is None or new is None:
            return {
                f"{prefix}_mae": float("nan"),
                f"{prefix}_max": float("nan"),
                f"{prefix}_max_dim": float("nan"),
                f"{prefix}_max_delta": float("nan"),
            }

        # 展平成一维，便于按维度号做索引与统计。
        old_arr = np.asarray(old, dtype=np.float64).reshape(-1)
        new_arr = np.asarray(new, dtype=np.float64).reshape(-1)
        # 两帧维度理论上一致；取较小者，避免越界（防御性写法）。
        dim = min(old_arr.shape[0], new_arr.shape[0])
        if indices is not None:
            # 只保留 indices 中"落在有效范围内"的维度（维度号必须 < dim）。
            valid_indices = [idx for idx in indices if idx < dim]
            if not valid_indices:
                # 指定的维度全部越界 → 无可统计维度，返回 NaN。
                return {
                    f"{prefix}_mae": float("nan"),
                    f"{prefix}_max": float("nan"),
                    f"{prefix}_max_dim": float("nan"),
                    f"{prefix}_max_delta": float("nan"),
                }
            old_arr = old_arr[valid_indices]
            new_arr = new_arr[valid_indices]
            # dim_indices 保存这些维度在原始动作向量里的真实编号，
            # 这样 max_dim 报告的是"第 7 维"而不是"过滤后的第 0 维"。
            dim_indices = np.asarray(valid_indices)
            dim = len(valid_indices)
        else:
            dim_indices = np.arange(dim)
        if dim == 0:
            return {
                f"{prefix}_mae": float("nan"),
                f"{prefix}_max": float("nan"),
                f"{prefix}_max_dim": float("nan"),
                f"{prefix}_max_delta": float("nan"),
            }
        # 逐维做差：delta = 新 - 旧。正值表示新帧比旧帧大，负值反之。
        delta = new_arr[:dim] - old_arr[:dim]
        abs_delta = np.abs(delta)
        # 找到"跳变最大"的维度及其带符号差值。
        max_pos = int(np.argmax(abs_delta))
        max_dim = int(dim_indices[max_pos])
        return {
            f"{prefix}_mae": float(np.mean(abs_delta)),
            f"{prefix}_max": float(abs_delta[max_pos]),
            f"{prefix}_max_dim": float(max_dim),
            f"{prefix}_max_delta": float(delta[max_pos]),
        }

    def _replace_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int):
        """RTC 模式：用新块**整体替换**旧队列，并按 delay 裁掉"已过去的帧"。

        逻辑：新块前 real_delay 帧对应的时间点已经过去（见模块 docstring
        第 4 点），直接丢掉；从第 real_delay 帧开始才是"未来"，存进队列。
        读指针复位到 0，从新队列开头逐帧消费。
        """
        # delay 可能越界（异常情况），安全夹到 [0, min(新旧块长度)]，避免切片越界。
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].copy()
        self.queue = processed_actions[clamped_delay:].copy()
        self.last_index = 0

        logger.debug(
            "replace: orig=%s, proc=%s, real_delay=%d, clamped=%d, "
            "remaining_orig=%s, remaining_proc=%s",
            original_actions.shape, processed_actions.shape,
            real_delay, clamped_delay,
            self.original_queue.shape, self.queue.shape,
        )

    def _append_actions_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray):
        """非 RTC 模式：把新块**追加**到队列尾部，维持动作连续性。

        逻辑：新块接在旧块后面继续执行。若旧队列为空，直接整块存下；
        否则拼接后，把已消费的前缀（前 last_index 行）裁掉、读指针复位，
        这样能持续执行而不丢帧，代价是动作可能有轻微重叠或平滑过渡需求。
        """
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            return

        # 拼接新块，再裁掉"已被 get() 消费掉的前缀"，把队首对齐到下一次要执行的位置。
        self.original_queue = np.concatenate([self.original_queue, original_actions.copy()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = np.concatenate([self.queue, processed_actions.copy()])
        self.queue = self.queue[self.last_index :]

        # 裁掉前缀后，队首就是下一个要执行的帧，读指针重新从 0 开始。
        self.last_index = 0

    def _check_and_resolve_delays(
        self,
        real_delay: int,
        action_index_before_inference: int | None = None,
        extra_delay: int = 0,
    ) -> int:
        """校验 delay 并返回"最终应该采用的裁帧量"。

        存在两个"时钟"测量 delay：
          - ``real_delay``：推理代码按墙钟估算的耗时（帧数）。
          - ``indexes_diff``：控制线程在推理期间**实际消费**的帧数
            （= 本次 merge 时的 last_index - 推理开始前的 last_index），
            即 RTC 论文里观测到的延迟 t - s。它由"读指针时钟"直接测得，
            与队列读取用的是同一把尺子，因此**优先采用**。

        解析规则：
          - 若提供了 action_index_before_inference，就用实际消费帧数
            （>= 0），再叠加 extra_delay（观测过期补偿）；两者不一致时按
            差异大小打印 debug / warning 日志，便于发现时钟不同步的问题。
          - 否则退回用 real_delay（夹到 >= 0）并加 extra_delay。
        """
        extra_delay = max(0, extra_delay)  # 补偿量不允许为负，防御性夹紧。
        effective_delay = max(0, real_delay) + extra_delay

        if action_index_before_inference is not None:
            # 推理期间实际消费的帧数（>=0）。
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            resolved = indexes_diff + extra_delay
            if indexes_diff != real_delay:
                # 两个时钟读数不一致：差异很小（<=1）可能是取整误差，用 debug；
                # 差异较大则用 warning 提醒排查（比如调用方传错了 real_delay）。
                log = logger.debug if abs(indexes_diff - real_delay) <= 1 else logger.warning
                log(
                    "Indexes diff != real delay. indexes_diff=%d, real_delay=%d, "
                    "extra_delay=%d, using=%d",
                    indexes_diff, real_delay, extra_delay, resolved,
                )
                return resolved
            return resolved

        # 没有提供"实际消费帧数"时，退回墙钟估算值。
        return effective_delay
