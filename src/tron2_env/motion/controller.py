"""MotionController — orchestrates transport + interpolator + publish loop.

**这个文件在整条链路里的位置（最重要的一点）：**
它是控制栈的**最顶层、总指挥**——环境（env）所有动作指令都通过它下达。
回顾完整链路（自底向上）：

    RobotTransport（传输）    → 只负责收发单帧，不懂"平滑"
    JointInterpolator（插值） → 把"当前关节角"平滑过渡到"目标关节角"
    MotionController（控制）  ← 本文件：把上面两者组装起来，并驱动一个发布线程

一句话概括 MotionController 的工作方式：
    1. start() 先读一次真实关节角，作为插值的起点（"对表"）；
    2. 启动一个**后台守护线程**，按 publish_rate（默认 300Hz）循环：
       每个节拍问插值器 current() 拿"此刻应该到哪"，发给 transport 推给机器人；
    3. 外部只调用 command_joints(target) 改插值器的目的地（**非阻塞**），
       线程会自动把机器人平滑带到新目标，并持续停在目标上。

This is the concrete object the environment talks to. It owns:

  * a :class:`RobotTransport`     — sends single frames to the robot
  * a :class:`JointInterpolator`  — produces smooth ``current(t)`` between waypoints
  * a daemon publish thread       — drives the transport at ``publish_rate`` Hz

The controller is transport-agnostic, but the public runtime currently exposes
the WebSocket transport only.

**为什么说"transport-agnostic（与传输方式无关）"？**
控制器只依赖 RobotTransport 接口（见 transport/base.py），不 care 底层到底是
websocket 还是内存假 transport。所以 examples/mock_quickstart.py 里可以塞一个
"只把指令记到列表里"的假 transport 来离线测试，完全不用改控制器。

Call ``command_joints(target)`` to retarget (non-blocking). The publish
thread keeps the robot at the latest target indefinitely thanks to PD on
the robot side.

**"持续停在目标上"是什么意思？**
机器人的伺服控制（servoj）是带 PD 位置环的：只要持续收到同一个目标角，
它就稳稳停在那里。所以发布线程哪怕每帧发的是同一个目标值也没关系，
机器人不会乱动——这保证了 RL 训练中"到达后保持位姿"的需求。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from tron2_env.config import Tron2Config
from tron2_env.errors import StateError
from tron2_env.interpolation import JointInterpolator, LinearInterpolator
from tron2_env.joints import JointIndex
from tron2_env.transport import RobotTransport, WebsocketTransport
from tron2_env.util import RateLimiter

logger = logging.getLogger(__name__)


class MotionController:
    """Owns a transport + interpolator + publish loop.

    持有"传输层 + 插值器 + 发布线程"三者，是整个控制栈的组装点。
    它自己不实现任何底层协议，只是把已有部件拼起来按固定节奏干活。
    """

    def __init__(
        self,
        transport: RobotTransport,
        interpolator: Optional[JointInterpolator] = None,
        publish_rate: float = 300.0,
        eta_default: float = 1.0 / 30.0,
    ) -> None:
        # ---- 四个核心字段 ----
        self._transport = transport
        # 插值器没传就用默认的 LinearInterpolator。注意类型注解是接口
        # JointInterpolator——换曲线插值类也不影响这里（见 interpolation/base.py）。
        self._interpolator = interpolator or LinearInterpolator()
        # 发布线程的运行频率（Hz）。默认 300Hz：每个节拍约 3.3ms 推一帧。
        self._publish_rate = publish_rate
        # command_joints 没给 eta 时用的默认"走完时间"（秒）。
        # 默认 1/30 ≈ 33ms，正好对应 30Hz 的控制周期（动作频率）。
        self._eta_default = eta_default

        # ---- 线程控制字段 ----
        # threading.Event：线程间"停机信号"。set() 后发布循环会退出。
        self._shutdown = threading.Event()
        # 发布线程句柄，disconnect 时 join 等它收尾。
        self._publish_thread: Optional[threading.Thread] = None
        # 是否已 start（防止重复启动 / 在未启动时误操作）。
        self._started = False

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Seed the interpolator with the current measured q, then launch the publish loop.

        先把"当前真实关节角"播种进插值器，再启动发布循环。

        Reading the first state before publishing is mandatory: otherwise the
        300 Hz loop would broadcast a default-initialised q and yank the robot.

        **为什么必须先读一次再发布？（新手最该理解的安全点）**
        如果跳过这一步，插值器内部是"默认全 0"状态，发布线程一启动就会把
        [0,0,...,0] 当成目标发出去——机器人会瞬间被拽到零位，非常危险。
        先 get_joint_state 读当前真实角度，reset 进插值器，循环才能从
        "当前位姿"出发、安全待命。
        """
        if self._started:
            # 幂等：已经启动过就直接返回，避免重复开线程。
            return
        try:
            # 读一次当前关节状态（阻塞等待，最多 2 秒）。
            # 注意：这一步在联网实时场景必须在**发布之前**完成。
            state = self._transport.get_joint_state(timeout=2.0)
        except StateError as exc:
            # 读不到状态 = 机器人没连上/没在出数据，直接升级成致命错误。
            # 用 raise ... from exc 保留原始异常链，方便排查。
            raise RuntimeError(
                "MotionController.start: could not read initial joint state — "
                "is the transport connected and producing state?"
            ) from exc

        # 状态是 18 维 [L_arm(7), L_grip(1), R_arm(7), R_grip(1), head(2)]，
        # 但伺服指令只要 16 维 [L_arm, R_arm, head]（两个夹爪不在里面）。
        # 这里做一次 18 → 16 的降维（见下方 _state_to_servoj）。
        q_now_18 = np.asarray(state["states"], dtype=np.float64)
        q_now_16 = self._state_to_servoj(q_now_18)
        # 播种：让插值器"当前即目的地"，发布循环启动后机器人保持不动、等待指令。
        self._interpolator.reset(q_now_16)

        # 启动发布线程。daemon=True：主程序退出时线程自动结束，不会被它卡住；
        # 但正式退出前请调用 disconnect() 做**优雅停机**，见其注释。
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name="MotionController-publish",
        )
        self._publish_thread.start()
        self._started = True
        logger.info(
            "MotionController started: publish_rate=%.1fHz eta_default=%.1fms",
            self._publish_rate,
            self._eta_default * 1000.0,
        )

    def disconnect(self) -> None:
        """优雅停机：先停发布线程，再断开传输层。

        顺序很重要：必须先发停机信号 + join 等线程退出，再 disconnect transport。
        否则可能线程还在发指令时底层连接已经被关掉，抛出一堆噪音异常。
        """
        self._shutdown.set()
        if self._publish_thread is not None and self._publish_thread.is_alive():
            # join(timeout=1.0)：最多等 1 秒让循环退出（它可能正卡在 send 上）。
            self._publish_thread.join(timeout=1.0)
        self._transport.disconnect()

    # ---- 支持 with 语句：`with MotionController(...) as mc:` ----
    # 进入 with 块返回 self；无论正常结束还是中途抛异常，__exit__ 都会调
    # disconnect()，保证资源一定被释放（类似文件对象的 with 用法）。
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ------------------------------------------------------------ public API

    def command_joints(
        self,
        target: np.ndarray,
        eta: Optional[float] = None,
    ) -> None:
        """Retarget. Non-blocking: just updates the interpolator destination.

        重新定向（非阻塞）：只改插值器的目的地，立刻返回。
        实际"走到目标"的动作由后台发布线程在后台完成。
        所以这里**可以高频调用**（比如 RL 里每个动作步调一次），
        线程会自动把新旧目标之间衔接成无折痕的轨迹（见 linear.py 的 kink-free）。
        """
        self._interpolator.set_destination(
            np.asarray(target, dtype=np.float64),
            # 没给 eta 就用默认值（≈1/fps）。给了就按指定时长走完。
            eta if eta is not None else self._eta_default,
        )

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        # 夹爪不是走插值链路的：它走 transport 的专用指令（0-100 开度）。
        # 因为夹爪是开关式/开度式控制，不需要平滑轨迹。
        self._transport.set_gripper(left_opening, right_opening)

    def get_joint_states(self, timeout: float = 1.0) -> dict:
        # 读当前关节状态（18 维，含夹爪）。直接转发给 transport。
        return self._transport.get_joint_state(timeout)

    def find_nearest_state(self, target_timestamp_s: float):
        """Find the queued state closest to target_timestamp_s (non-destructive).

        在传输层缓存的状态队列里，找与目标时刻最接近的那一帧状态。

        Returns ``None`` if the underlying transport doesn't support search or
        if the queue is empty. Useful for image/state alignment in legacy obs.

        **实现技巧：getattr 探测方法是否存在（鸭子类型的运行时版）。**
        只有部分 transport（如 WebsocketTransport）实现了 find_nearest_state。
        这里不 isinstance 判断、也不硬调方法，而是用 getattr 拿到方法；
        没有就返回 None。这样 mock transport（测试用）即便没实现这个功能，
        控制器也不会崩——优雅降级。
        """
        finder = getattr(self._transport, "find_nearest_state", None)
        if finder is None:
            return None
        return finder(target_timestamp_s)

    def get_head_position(self) -> np.ndarray:
        # 读头部位姿（2 维：pitch / yaw），转发给 transport。
        return self._transport.get_head_position()

    def wait_until_reached(self, *args, **kwargs) -> bool:
        # 阻塞等待"当前目标运动完成"（transport 侧用 14 维手臂位姿判断，
        # 超时返回 False）。*args/**kwargs 直接透传——控制器不关心细节。
        return self._transport.wait_until_reached(*args, **kwargs)

    def is_connected(self) -> bool:
        return self._transport.is_connected()

    @property
    def transport(self) -> RobotTransport:
        """Exposes the underlying transport for backend-specific operations (e.g. movej during reset).

        把底层 transport 暴露给外部，用于走通用接口覆盖不到的**后端专属操作**。
        典型场景：reset 复位时需要"整段轨迹移动"（movej），这不是逐帧伺服
        命令能表达的，只能绕过通用接口直接调用 websocket 特有的 movej 方法。
        这也是设计上留的口子：通用方法走控制器，特殊方法走 .transport。
        """
        return self._transport

    # ----------------------------------------------------------------- loop

    def _publish_loop(self) -> None:
        """发布主循环：每个节拍"取插值 → 发指令"。

        这是后台线程的入口，start() 之后一直运行，直到收到停机信号。
        """
        # RateLimiter：固定频率的节拍器（见 util.py）。它基于单调时钟，
        # 即使某次循环超时了，也会自动"追回"节奏而不累积相位漂移。
        rate = RateLimiter(self._publish_rate)
        while not self._shutdown.is_set():
            try:
                # 取"此刻应到的关节角"（16 维），发给机器人。
                q = self._interpolator.current()
                self._transport.send_joint_cmd(q)
            except Exception as exc:  # noqa: BLE001
                # 单次发送失败不能杀死整个循环：
                #   插值器仍持有最新的目标，下个节拍重试即可。
                # 这也意味着瞬时网络抖动不会中断控制，只丢一帧。
                logger.warning("publish loop: %s", exc)
            rate.sleep()  # 睡到下一个节拍时间点（≈ 1/publish_rate 秒）
            # （while 的条件会立即再检查 shutdown，所以停机是及时的）

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _state_to_servoj(state18: np.ndarray) -> np.ndarray:
        """Convert an 18-dim ``[L_arm, L_grip, R_arm, R_grip, head]`` state
        into the 16-dim ``[L_arm, R_arm, head]`` setpoint vector.

        状态（18 维）→ 伺服目标（16 维）的降维映射。两者布局见 joints.py：
            STATE_DIM=18:  [L_arm(7) | L_grip(1) | R_arm(7) | R_grip(1) | head(2)]
            SERVOJ_DIM=16: [L_arm(7) | R_arm(7) | head(2)]
        丢掉两个夹爪维度，因为夹爪不在 servoj 指令里（夹爪走 set_gripper）。
        用 JointIndex 的切片常量（LEFT_ARM/RIGHT_ARM/HEAD）取段，
        避免手写魔法下标 [0:7]、[8:15] 之类。
        """
        return np.concatenate(
            [
                state18[JointIndex.LEFT_ARM],
                state18[JointIndex.RIGHT_ARM],
                state18[JointIndex.HEAD],
            ]
        )


# ---------------------------------------------------------------------- factory


def create_motion_controller(
    config: Tron2Config,
    backend: str = "websocket",
    publish_rate: float = 300.0,
    eta_default: float = 1.0 / 30.0,
    interpolator: Optional[JointInterpolator] = None,
) -> MotionController:
    """Build a started MotionController with the requested backend.

    工厂函数：一行代码拿到一个**已经启动好**的控制器。
    内部做了"建 transport → 组装控制器 → start() 读初值并开发布线程"三件事，
    省得调用方自己一步步搭。

    Args:
        config: Robot connection + bring-up params.
            Tron2Config 里含 robot_ip / port（连接参数）、init_joints(14维手臂) /
            init_head(2维) / init_ee_z_min（上电初始位姿与安全参数）、
            state_queue_maxlen / polling_rate / connection_timeout（传输层调参），
            字段校验在 dataclass 的 __post_init__ 里（见 config.py）。
        backend: ``"websocket"``.
            目前唯一支持的后端。留这个参数是为了将来能加别的传输实现。
        publish_rate: Hz at which the publish loop drives the transport.
            发布线程频率（默认 300Hz）。
        eta_default: default time-to-target for ``command_joints`` (= 1/fps).
            默认走完目标的时间（默认 ≈33ms，对应 30Hz 动作频率）。
        interpolator: override the default ``LinearInterpolator``.
            自定义插值器（可换曲线算法），默认线性。
    """
    if backend != "websocket":
        raise ValueError(f"unknown control backend: {backend!r} (expected 'websocket')")
    # 创建真正连机器人的 websocket transport（见 transport/websocket.py）。
    transport: RobotTransport = WebsocketTransport(config)

    mc = MotionController(
        transport=transport,
        interpolator=interpolator,
        publish_rate=publish_rate,
        eta_default=eta_default,
    )
    # 关键：这里调用 start()——连接、读初值、开发布线程，返回即"可用"。
    mc.start()
    return mc
