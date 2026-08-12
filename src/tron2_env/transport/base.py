"""RobotTransport abstraction — the bottom layer of the control stack.

这是整个控制栈的**最底层抽象**，相当于整条控制链路的"地基"。
读本文件前请先建立一个整体认识（建议配合 examples/mock_quickstart.py 一起看）：

    MotionController（控制层，负责节奏）
        │  持有 transport
        ▼
    RobotTransport（本文件，传输层，只管"收发单帧"）
        │  真正连到机器人的实现
        ▼
    机器人本体（ws://<ip>:5000 或假机器人）

传输层只暴露**单发单收**的原语：

  * ``send_joint_cmd(q[16])``                — 发一条关节指令（16 维）
  * ``get_joint_state(timeout)``             — 读最新 18 维状态
  * ``get_head_position()``                  — 读最新头部 [pitch, yaw]
  * ``set_gripper(left, right)``             — 驱动夹爪
  * ``wait_until_reached(target, tol, t)``   — 阻塞轮询，等机械臂到位
  * ``disconnect`` / ``is_connected``        — 生命周期管理

它**不做**限频（rate limiting）、**不做**插值（interpolation）、**没有**
后台发布循环（publish loop）——这些职责全部在上层
``tron2_env.motion.MotionController`` 里。

为什么要把"传输"和"节奏"拆开？三个好处：
  1. 传输层保持"极简"，自然就非常容易测试——可以用假连接（内存版）来测；
  2. 同一个发布循环可以驱动任何后端（真机器人 / 模拟 / 录制回放），只需换 transport；
  3. 上层代码只依赖这个接口，不关心底层到底走什么协议。

具体实现：
  * ``tron2_env.transport.websocket.WebsocketTransport`` — JSON over ws://...
    注意：它**并没有继承本类**，只是实现了同样的方法（见下方对 Protocol 的说明）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# 为什么这里用 Protocol，而不是普通的抽象基类（ABC）？
# ---------------------------------------------------------------------------
# Protocol 是 Python 3.8+ 从 typing 模块引入的"结构子类型（structural
# subtyping）"机制，也叫 **鸭子类型（duck typing）的静态版本**：
#
#   不要求某个类显式声明 "我继承自 RobotTransport"，
#   只要它实现了这里定义的全部方法（长着一样的鸭子），
#   在类型检查层面它就算 RobotTransport。
#
# 传统写法（ABC + @abstractmethod）是"名义子类型（nominal subtyping）"：
#   必须写出 `class WebsocketTransport(RobotTransport)` 才被承认是子类。
# 但本仓库的实际实现 WebsocketTransport 并没有这么做——
# 两者之所以能通用，正是靠 Protocol 的这种"按结构匹配"规则。
#
# 直观理解：
#   - 定义端（本文件）："想要当 transport，你得会这些动作。"
#   - 使用端（MotionController / env）："我不管你是谁，会这些动作就行。"
#   - 实现端（WebsocketTransport / RecordingTransport）：各自实现方法，
#     无需知道协议的存在。
#
# 好处：
#   1. 不必为测试造"假子类"，随便写个只有相同方法的类就能顶替；
#   2. 没有继承时的耦合，实现类可以自由地拥有自己的额外方法；
#   3. 给 IDE / mypy / pyright 提供类型提示：MotionController 收到的是
#      "实现了这套接口的对象"，从而可以静态检查方法调用是否有误。
# ---------------------------------------------------------------------------

# @runtime_checkable 让"结构子类型"从纯静态扩展到运行时：
# 加上它之后，`isinstance(obj, RobotTransport)` 会在运行时检查 obj 是否
# 实现了协议里声明的全部方法（而没有它时 isinstance 总是返回 False）。
# 这在调试 / 断言"传入的对象确实是 transport"时很方便。
@runtime_checkable
class RobotTransport(Protocol):
    """Single-shot send/recv primitives. No internal pacing.

    单发单收原语的接口契约。所有方法都刻意保持"一次一帧"、不带节拍。
    """

    def send_joint_cmd(self, q: np.ndarray) -> None:
        """Send a 16-dim joint setpoint [L_arm(7), R_arm(7), head(2)].

        发送一条 16 维的关节目标指令（弧度），布局见 JointIndex.SERVOJ_DIM。
        注意：**夹爪不在这里面**——夹爪由独立的 set_gripper() 控制。

        Non-blocking; the actual on-the-wire send may be async. Raises
        ``CommandError`` if the transport is disconnected or the array is
        the wrong shape.

        非阻塞：调用后立即返回，真正把数据写到网线上可能是异步的。
        若连接已断开、或数组维度不对，抛出 ``CommandError``。
        """

    def get_joint_state(self, timeout: float = 1.0) -> dict:
        """Return the latest 18-dim state.

        读取机器人最新的一帧 18 维状态（布局见 JointIndex.STATE_DIM）：

        Shape::

            {
                "timestamp": int (ms),      # 客户端本地墙钟时间（毫秒）
                "states": list[float] (18), # [L_arm(7), L_grip, R_arm(7), R_grip, head(2)]
                ...
            }

        语义：这是一个"拉取"接口——调用方主动来取最新状态。
        在 WebSocket 实现里，内部有后台线程不断把机器人上报的状态
        压进一个队列，这里只是从队列里取一帧（并可阻塞等待最多 timeout 秒）。

        Raises ``StateError`` on timeout.
        超时（timeout 秒内没有新状态）抛出 ``StateError``。
        """

    def get_head_position(self) -> np.ndarray:
        """Return the latest 2-dim head [pitch, yaw] without dequeuing state.

        返回最新头部关节角 [pitch(俯仰), yaw(偏航)]，2 维。
        和 get_joint_state 的关键区别：**不消耗**状态队列（不取走帧），
        只是一个"窥探"式的便捷读取。
        """

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        """Set gripper opening 0..100 for left/right.

        设置左右夹爪开合度，取值 0~100（0=闭合，100=全开）。
        夹爪不进 16 维关节指令，而是用这个独立命令直接驱动。
        """

    def wait_until_reached(
        self,
        target_joints,
        tolerance: float = 0.05,
        timeout: float = 10.0,
    ) -> bool:
        """Block until arm joints are within ``tolerance`` rad of ``target_joints`` (14-dim).

        阻塞式轮询：每隔一小段时间读一次状态，直到左右臂（14 维，
        不含夹爪和头部）所有关节与 target_joints 的误差都小于 tolerance
        （弧度）才返回 True；超时则返回 False。
        常用于 env.reset 时"等机器人真的走到初始位姿"。
        """

    def disconnect(self) -> None:
        """Close the connection and stop any internal threads.

        关闭连接，并停掉实现内部可能存在的线程（如状态轮询线程）。
        调用后 is_connected() 应返回 False。必须幂等（多次调用无副作用）。
        """

    def is_connected(self) -> bool: ...

    # ------------------------------------------------------------------
    # 下面两个方法让 transport 支持 with 语句：
    #   with transport as t:
    #       ...  # 块结束自动调用 disconnect()
    # 这样即使中途抛异常也能保证连接被正确释放。
    # ------------------------------------------------------------------
    def __enter__(self): ...

    def __exit__(self, exc_type, exc_val, exc_tb): ...
