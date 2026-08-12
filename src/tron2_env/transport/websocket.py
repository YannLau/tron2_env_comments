"""WebsocketTransport — high-level ws://<ip>:5000 JSON control.

这是 ``RobotTransport`` 协议（见 transport/base.py）的**真实实现**——
它通过 WebSocket 直连真实机器人，用 JSON 消息收发控制指令与状态。

阅读本文件前，请先记住两个"心智模型"：

**1) 请求/响应模型（像点菜）**
所有指令都走同一个模式：发送方构造一条 JSON 消息（含 title 字段），
机器人处理完回一条 JSON 响应（title 为 "response_xxx"），收到后按 title 路由。

    我们发出        request_servoj / request_get_joint_state / request_movej ...
    机器人响应      response_servoj / response_get_joint_state / response_movej ...

**2) 双线程模型**
本类内部跑着两个后台线程（都是 daemon，不阻止程序退出）：
  * websocket 收包线程  —— 阻塞在 ws_client.run_forever() 上，被动接收机器人
    随时可能推送的任何消息，按 title 分发到对应处理函数。
  * 状态轮询线程        —— 主动、周期性（polling_rate Hz）发请求，索取
    "关节状态"和"夹爪状态"。
两者用"线程安全队列 + 锁"衔接：收包线程把新状态写进队列，
上层 get_joint_state() 同步地从队列里取。

Migrated from the original ``tron2_env.robot.Tron2`` class. Changes vs. the
old implementation:

* Renamed to make the abstraction layer explicit (``RobotTransport``).
* Single-shot ``send_joint_cmd`` replaces public ``servoj``; the internal
  ``servoj_rate_limiter`` is removed. Pacing now lives in
  ``MotionController._publish_loop``.
* Added ``get_head_position`` so callers no longer reach into private state.
* Dropped methods with zero production callers: ``movep``, ``servop``,
  ``chassis_*``, ``lifter_*``, ``set_light_effect``, ``emergency_stop``.
  The corresponding ws subscriptions and state buffers are gone too.
* ``movej`` / ``move_head`` / ``wait_until_reached`` / ``set_gripper``
  retained — used by ``_move_to_init_pose`` and ``env.reset``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from typing import Dict, List, Optional, Union

import numpy as np
import websocket

from tron2_env.config import Tron2Config
from tron2_env.errors import CommandError, StateError
from tron2_env.joints import JointIndex


class WebsocketTransport:
    """Tron2 control via the ws://<ip>:5000 JSON protocol.

    Implements :class:`tron2_env.transport.base.RobotTransport`.

    注意：这里写的是 "Implements"（实现）而不是 "Inherits"（继承）——
    这个类**并没有** `class WebsocketTransport(RobotTransport)`。
    之所以仍然被认为是合法 transport，全靠 base.py 里 RobotTransport 是
    ``Protocol``：只要方法签名齐全，结构上就匹配（鸭子类型的静态化）。
    所以本文件只要把接口方法实现全，上层 MotionController 就能直接使用。

    Example:
        >>> config = Tron2Config(robot_ip="ROBOT_IP")
        >>> with WebsocketTransport(config) as t:
        ...     t.send_joint_cmd(np.zeros(16))
        ...     state = t.get_joint_state()
    """

    # 16 维控制指令的长度常量，直接复用 JointIndex 里的定义，
    # 避免各处在写死 16 这个魔法数字。
    SERVOJ_DIM = JointIndex.SERVOJ_DIM

    def __init__(self, config: Optional[Tron2Config] = None) -> None:
        self.config = config or Tron2Config()
        self._setup_logger()

        # ws connection —— 连接相关状态
        self.accid: Optional[str] = None             # "账户/会话 ID"，握手后由机器人下发，
                                                     # 之后所有请求都要带回去，用于身份关联
        self.ws_client: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.connected = False
        self.should_exit = False                     # 退出标记，轮询线程用它判断何时停

        # state buffers
        # 说明：机器人把"关节"和"夹爪"分成两条独立消息上报，
        # 因此下面用两把锁 + 两个"最新一帧"缓存来暂存最近收到的部分状态。
        # joint_states / ee_pose_states = 各类型"最近收到的一帧"
        # xxx_queue = 已凑齐/已封装好的完整帧的队列（双端队列，有长度上限）
        # timestamp = 客户端 wall clock ms（本地时间），
        # robot_timestamp = 机器人自带的服务器时间戳（仅调试用）
        self.joint_states: Dict = {
            "timestamp": -1,
            "robot_timestamp": -1,
            "states": [-1.0] * JointIndex.STATE_DIM,
            "joint_updated": False,     # 本帧是否已收到"关节"部分
            "gripper_updated": False,   # 本帧是否已收到"夹爪"部分
        }
        self.ee_pose_states: Dict = {
            "timestamp": -1,
            "left_position": [-1.0, -1.0, -1.0],   # 末端位置 (x, y, z)
            "left_quat": [-1.0, -1.0, -1.0, -1.0], # 末端姿态（四元数）
            "right_position": [-1.0, -1.0, -1.0],
            "right_quat": [-1.0, -1.0, -1.0, -1.0],
        }
        # deque(maxlen=...) 是"环形缓冲"：满了自动挤掉最旧的，天然保留最近 N 帧。
        self.joint_state_queue: deque = deque(maxlen=self.config.state_queue_maxlen)
        self.ee_pose_queue: deque = deque(maxlen=self.config.state_queue_maxlen)
        self._queue_lock = threading.Lock()   # 保护上面两个队列
        self._state_lock = threading.Lock()   # 保护 joint_states / ee_pose_states

        # bring-up —— 构造时就立即连接，并完成初始化动作
        self._connect()                        # 1. 建立 websocket，启动收包线程
        time.sleep(0.1)
        self._start_polling_thread()           # 2. 启动状态轮询线程
        # 3. 若配置了初始位姿，就把机器人挪过去（安全起见，
        #    有时要"先绕一个中间位姿再走"，见 _move_to_init_pose）。
        # 这是一个预置的"中间过渡位姿"（14 维，仅左右臂），
        # 当机器人离初始位太近时，用它绕开奇异路径（见 _joint_pose_needs_second_joint）。
        self._second_joint = [
            0.000999913, -0.00449967, 1.482, -1.57, 0.0036, 0.00289989, -0.00160009,
            0.0415001, 0.1279, -1.4808, -1.57, -0.00739986, 0.0151, -0.0624998,
        ]

        if self.config.init_joints is not None or self.config.init_head is not None:
            self._move_to_init_pose()

    # ------------------------------------------------------------------ logger

    def _setup_logger(self) -> None:
        # 每个连接实例一个独立 logger，名字里带机器人 IP，方便多机器人时区分日志。
        self.logger = logging.getLogger(f"WebsocketTransport-{self.config.robot_ip}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.propagate = False   # 不向上传播，避免和根 logger 重复打印

    # --------------------------------------------------------------- websocket

    @staticmethod
    def _new_guid() -> str:
        # 每条请求一个唯一 ID，方便在日志/机器人侧跟踪单条消息。
        return str(uuid.uuid4())

    def _send_request(self, title: str, data: Optional[Dict] = None) -> bool:
        # 所有外发指令的唯一入口：把"业务参数 + 通用壳"打包成一条 JSON 消息发出去。
        # 消息结构：{accid, title, timestamp, guid, data}
        #   - title 决定"这条消息是干嘛的"（机器人按它路由）
        #   - data  是具体参数（如 {"q": [...]}）
        if not self.ws_client or not self.connected:
            self.logger.warning("ws not connected, dropping: %s", title)
            return False
        try:
            message = {
                "accid": self.accid,
                "title": title,
                "timestamp": int(time.time() * 1000),
                "guid": self._new_guid(),
                "data": data or {},
            }
            self.ws_client.send(json.dumps(message))
            return True
        except Exception as exc:
            self.logger.error("send failed (%s): %s", title, exc)
            return False

    def _on_open(self, ws) -> None:
        # websocket 连接成功的回调。置 connected=True，之后才能发请求。
        self.logger.info("connected: %s:%s", self.config.robot_ip, self.config.port)
        self.connected = True

    def _on_message(self, ws, message: str) -> None:
        # websocket 每收到一条消息都会触发这里（由收包线程回调）。
        # 收到的都是 JSON 字符串，解析后按 title 分发到对应处理函数——
        # 这就是"请求/响应模型"里的响应侧。
        try:
            root = json.loads(message)
            title = root.get("title", "")
            self.accid = root.get("accid", self.accid)  # 顺手记住机器人给的会话 ID
            if title == "response_get_joint_state":
                self._handle_joint_state(root)
            elif title == "response_get_limx_2fclaw_state":
                self._handle_gripper_state(root)
            elif title == "response_get_move_pose":
                self._handle_ee_pose(root)
            elif title not in {
                # 这些属于"低频/一次性"响应，无需更新状态缓存，静默忽略即可
                "notify_robot_info",
                "response_servoj",
                "response_set_limx_2fclaw_cmd",
                "response_movej",
                "response_moveh",
            }:
                self.logger.debug("rx: %s", title)
        except json.JSONDecodeError:
            self.logger.error("bad json: %s", message)
        except Exception as exc:
            self.logger.error("on_message error: %s", exc)

    def _handle_joint_state(self, root: Dict) -> None:
        # 处理"关节状态响应"：把机器人发来的 q 数组（16 维，仅关节，无夹爪）
        # 填进 18 维 state 里对应的切片，头部从第 14/15 个元素取。
        data = root.get("data", {})
        joint_q = data.get("q", [])
        # 用客户端 wall clock 作为统一时间基准 (与 camera.py 中的 time.time() 一致)
        # 机器人原始时间戳保留在 robot_timestamp 字段供调试。
        client_ts_ms = int(time.time() * 1000)
        with self._state_lock:
            self.joint_states["timestamp"] = client_ts_ms
            self.joint_states["robot_timestamp"] = root.get("timestamp", -1)
            # q 的布局是 [L_arm(7), R_arm(7), head(2)]，与 JointIndex 完全对应，
            # 于是用切片分别填进 18 维 state 对应位置：
            self.joint_states["states"][JointIndex.LEFT_ARM] = joint_q[: JointIndex.ARM_JOINT_DIM]
            self.joint_states["states"][JointIndex.RIGHT_ARM] = joint_q[
                JointIndex.ARM_JOINT_DIM : JointIndex.TOTAL_ARM_DIM
            ]
            self.joint_states["states"][JointIndex.HEAD_PITCH] = joint_q[14]
            self.joint_states["states"][JointIndex.HEAD_YAW] = joint_q[15]
            self.joint_states["joint_updated"] = True   # 本帧"关节部分"已到位
            self._try_commit_state()                    # 尝试凑成完整帧入队

    def _handle_gripper_state(self, root: Dict) -> None:
        # 处理"夹爪状态响应"。机器人回报的开合度是 0~100 的百分比，
        # 这里 /100 归一化成 0~1（和关节角同一数量级，方便一起处理）。
        data = root.get("data", {})
        with self._state_lock:
            self.joint_states["states"][JointIndex.LEFT_GRIPPER] = data.get("left_opening", -1) / 100.0
            self.joint_states["states"][JointIndex.RIGHT_GRIPPER] = data.get("right_opening", -1) / 100.0
            self.joint_states["gripper_updated"] = True  # 本帧"夹爪部分"已到位
            self._try_commit_state()                     # 尝试凑成完整帧入队

    def _handle_ee_pose(self, root: Dict) -> None:
        # 处理"末端位姿响应"（仅当主动请求 move_pose 时才会收到）。
        # 末端的 6 自由度信息（位置 + 四元数姿态）单独存一份，不入关节状态。
        data = root.get("data", {})
        pose = {
            "timestamp": data.get("timestamp", root.get("timestamp", -1)),
            "left_position": data.get("left_position", [-1.0, -1.0, -1.0]),
            "left_quat": data.get("left_quat", [-1.0, -1.0, -1.0, -1.0]),
            "right_position": data.get("right_position", [-1.0, -1.0, -1.0]),
            "right_quat": data.get("right_quat", [-1.0, -1.0, -1.0, -1.0]),
            "result": data.get("result"),
        }
        self.ee_pose_states = pose
        with self._queue_lock:
            self.ee_pose_queue.append(pose.copy())

    def _try_commit_state(self) -> None:
        """Commit to queue only when both joint and gripper frames have arrived.

        状态"合成门闩"：因为关节和夹爪是两条独立消息到达的，任何一个先到
        都不构成完整状态。只有当**两者都到了**（joint_updated 和 gripper_updated
        同时为 True），且数据有效（机械臂值不是初始的 -1、时间戳正常），
        才把这一帧完整 18 维状态压入队列，然后清掉两个"已更新"标记，
        等待下一对消息。

        这样上层 get_joint_state() 从队列里取到的永远是"完整的"状态帧。
        """
        if (
            self.joint_states["joint_updated"]
            and self.joint_states["gripper_updated"]
            and self.joint_states["states"][JointIndex.LEFT_ARM_START] != -1
            and self.joint_states["timestamp"] != -1
        ):
            with self._queue_lock:
                self.joint_state_queue.append(self.joint_states.copy())
            self.joint_states["joint_updated"] = False
            self.joint_states["gripper_updated"] = False

    def _on_close(self, ws, status_code, msg) -> None:
        # 连接被关闭的回调：置 connected=False，之后发请求会被拒。
        self.logger.warning("ws closed: %s - %s", status_code, msg)
        self.connected = False

    def _on_error(self, ws, error) -> None:
        self.logger.error("ws error: %s", error)

    def _connect(self) -> None:
        # 建立连接：构造 WebSocketApp，注册回调，然后起一个线程跑 run_forever()。
        url = f"ws://{self.config.robot_ip}:{self.config.port}"
        self.logger.info("connecting: %s", url)
        self.ws_client = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
        )
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()

    def _run_websocket(self) -> None:
        # 收包线程的主体：run_forever() 会**阻塞式**地持续收包，
        # 每收到一条消息自动回调 _on_message；连接断开则返回。
        try:
            self.ws_client.run_forever()
        except Exception as exc:
            self.logger.error("ws thread crashed: %s", exc)

    # ---------------------------------------------------------------- polling

    def _start_polling_thread(self) -> None:
        # 启动轮询线程，按 config.polling_rate 周期性索要状态。
        thread = threading.Thread(target=self._poll_feedback, daemon=True)
        thread.start()
        self.logger.info("state polling started (%s Hz)", self.config.polling_rate)

    def _poll_feedback(self) -> None:
        # 轮询线程主体：不停发"关节状态"和"夹爪状态"两个请求。
        # 响应异步回来，最终落到 joint_state_queue（见 _try_commit_state）。
        period = 1.0 / self.config.polling_rate
        while not self.should_exit:
            t0 = time.time()
            self._send_request("request_get_joint_state")
            self._send_request("request_get_limx_2fclaw_state")
            # 精确控频：睡到下一个周期（扣除本次请求耗时），避免 drift。
            time.sleep(max(0.0, period - (time.time() - t0)))

    # ----------------------------------------------------------- RobotTransport

    def send_joint_cmd(self, q: Union[List[float], np.ndarray]) -> None:
        """Send one 16-dim servoj setpoint (no rate limiting — caller paces).

        实现 RobotTransport 接口。只发一帧 servoj 指令，**不做限频**——
        调频是上层 MotionController._publish_loop 的职责（这正是接口设计
        刻意把两者分开的原因，见 base.py）。这里只负责：
        类型转换（numpy → list）、维度校验、打包发送。
        """
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if len(q) != self.SERVOJ_DIM:
            raise CommandError(f"expected {self.SERVOJ_DIM}-dim q, got {len(q)}")
        if not self._send_request("request_servoj", {"q": q, "filter_ratio": 1.0}):
            raise CommandError("servoj send failed")

    def get_joint_state(self, timeout: float = 1.0) -> Dict:
        # 从"完整状态队列"里取一帧。队列里有就立刻取走（先进先出），
        # 没有就每 1ms 轮询一次，直到超时抛 StateError。
        start = time.time()
        while time.time() - start < timeout:
            with self._queue_lock:
                if self.joint_state_queue:
                    states = self.joint_state_queue.popleft()
                    if len(states["states"]) != JointIndex.STATE_DIM:
                        raise StateError(
                            f"bad state dim: expected {JointIndex.STATE_DIM}, got {len(states['states'])}"
                        )
                    return states
            time.sleep(0.001)
        raise StateError(f"get_joint_state timed out ({timeout}s)")

    def get_ee_poses(self, timeout: float = 1.0) -> Dict:
        """Request and return the latest end-effector poses.

        末端位姿不是周期性上报的，需要"主动请求一次、等一次响应"。
        做法：先清空队列，发一条请求，再在 timeout 内等队列里出现一帧。
        """
        with self._queue_lock:
            self.ee_pose_queue.clear()
        if not self._send_request("request_get_move_pose"):
            raise StateError("get_ee_poses request failed")

        start = time.time()
        while time.time() - start < timeout:
            with self._queue_lock:
                if self.ee_pose_queue:
                    pose = self.ee_pose_queue.popleft()
                    result = pose.get("result")
                    if result not in (None, "success"):
                        raise StateError(f"get_ee_poses failed: {result}")
                    return pose
            time.sleep(0.001)
        raise StateError(f"get_ee_poses timed out ({timeout}s)")

    def find_nearest_state(self, target_timestamp_s: float) -> Optional[Dict]:
        """Find the state in the queue whose timestamp is closest to target.

        Non-destructive: does NOT consume the queue. Returns None if the queue
        is empty. The timestamp field in state dicts is in milliseconds.

        用途：把"某一时刻的图像"和"最近的关节状态"对齐（图像/状态同步）。
        关键点：**只读不改**——不弹出任何帧，只在队列里找出时间戳最接近的
        那一帧并返回其副本。目标时间以秒传入，队内时间戳是毫秒，先换算。
        """
        with self._queue_lock:
            if not self.joint_state_queue:
                return None
            target_ms = target_timestamp_s * 1000.0
            best = None
            best_diff = float("inf")
            for state in self.joint_state_queue:
                diff = abs(state["timestamp"] - target_ms)
                if diff < best_diff:
                    best_diff = diff
                    best = state
            return best.copy() if best is not None else None

    def get_head_position(self) -> np.ndarray:
        """Return latest head [pitch, yaw] without dequeuing state.

        头部 [pitch, yaw] 的轻量读取：直接从"最近一帧"缓存里切片取，
        不碰队列（即不消费状态）。若缓存还没填满就返回 [0, 0] 兜底。
        """
        with self._state_lock:
            states = self.joint_states["states"]
            if len(states) < JointIndex.STATE_DIM:
                return np.array([0.0, 0.0])
            return np.array(states[JointIndex.HEAD], dtype=np.float64)

    def set_gripper(
        self,
        left_opening: float = 0.0,
        right_opening: float = 0.0,
        left_speed: float = 100.0,
        left_force: float = 50.0,
        right_speed: float = 100.0,
        right_force: float = 50.0,
    ) -> None:
        # 夹爪独立控制（不进 16 维 servoj 指令）。
        # np.clip(x, 0, 100) 把输入限制在 [0,100]（防御非法值），再转 int。
        data = {
            "left_opening": int(np.clip(left_opening, 0, 100)),
            "left_speed": int(np.clip(left_speed, 0, 100)),
            "left_force": int(np.clip(left_force, 0, 100)),
            "right_opening": int(np.clip(right_opening, 0, 100)),
            "right_speed": int(np.clip(right_speed, 0, 100)),
            "right_force": int(np.clip(right_force, 0, 100)),
        }
        self._send_request("request_set_limx_2fclaw_cmd", data)

    def wait_until_reached(
        self,
        target_joints: Union[List[float], np.ndarray],
        tolerance: float = 0.05,
        timeout: float = 10.0,
    ) -> bool:
        # 阻塞轮询直到左右臂到位（或超时）。
        # 实现思路很朴素：循环里反复读状态，算"当前 - 目标"的最大绝对误差，
        # 小于 tolerance 即成功返回；读状态偶尔失败（StateError）则重试；
        # 超过 timeout 返回 False（不抛异常，让调用方决定怎么处理）。
        if isinstance(target_joints, np.ndarray):
            target_joints = target_joints.tolist()
        target = np.array(target_joints)
        start = time.time()
        while time.time() - start < timeout:
            try:
                states = self.get_joint_state(timeout=1.0)
                current = states["states"]
                # 只比左右臂（14 维，去掉夹爪和头部）
                arm = np.array(current[JointIndex.LEFT_ARM] + current[JointIndex.RIGHT_ARM])
                diff = arm - target
                err = float(np.max(np.abs(diff)))
                if err < tolerance:
                    self.logger.info("reached target (err=%.4f)", err)
                    return True
                time.sleep(0.1)   # 每 100ms 检查一次，避免空转烧 CPU
            except StateError:
                self.logger.warning("state read failed; retrying")
                continue
        self.logger.warning("wait_until_reached timed out (%.1fs)", timeout)
        return False

    def is_connected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        # 断开连接：置退出标记（让轮询线程停）、关掉 websocket、等收包线程退出。
        # 注意 __del__ 里也调它，防止对象被回收时连接泄漏。
        self.logger.info("disconnecting")
        self.should_exit = True
        if self.ws_client:
            self.ws_client.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2.0)
        self.connected = False
        self.logger.info("disconnected")

    def __enter__(self):
        # 支持 with 语法：进入时返回自身。
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # with 块结束自动断开连接（无论是否抛异常）。
        self.disconnect()

    def __del__(self):
        # 对象被垃圾回收时的最后兜底：若还连着就断开，防止资源泄漏。
        # getattr 防御：构造中途失败时可能还没有 connected 属性。
        if getattr(self, "connected", False):
            try:
                self.disconnect()
            except Exception:
                pass

    # ----------------------------------------------------- ws-specific helpers
    # These are NOT part of RobotTransport. They are used by ``_move_to_init_pose``
    # and the environment's reset path.
    # 注意：下面的方法**不属于** RobotTransport 接口，是 WebSocket 实现独有的
    # "额外能力"。上层通过 controller.transport 访问时才能用到。
    # 这些是"一整段轨迹"级别的指令，由机器人侧控制器自己完成插值；
    # 而 send_joint_cmd 是"逐帧"级别的，由本仓库的插值器负责平滑。

    def movej(self, joint_positions: Union[List[float], np.ndarray], move_time: float = 2.0) -> None:
        """Interpolated joint-space motion (ws controller does the trajectory).

        关节空间插值运动：告诉机器人"用 move_time 秒，从当前走到 joint_positions
        （14 维，仅左右臂）"，轨迹由机器人侧控制器插值，本端不逐帧发指令。
        """
        if isinstance(joint_positions, np.ndarray):
            joint_positions = joint_positions.tolist()
        if len(joint_positions) != JointIndex.MOVEJ_DIM:
            raise CommandError(f"expected {JointIndex.MOVEJ_DIM}-dim joints, got {len(joint_positions)}")
        if not self._send_request("request_movej", {"joint": joint_positions, "time": move_time}):
            raise CommandError("movej send failed")
        self.logger.debug("movej sent (time=%.2fs)", move_time)

    def move_head(self, head_joint: Union[List[float], np.ndarray], move_time: float = 5.0) -> None:
        """Interpolated head motion.

        头部关节的插值运动（2 维 [pitch, yaw]），同理由机器人侧插值。
        """
        if isinstance(head_joint, np.ndarray):
            head_joint = head_joint.tolist()
        if len(head_joint) != JointIndex.HEAD_DIM:
            raise CommandError(f"expected {JointIndex.HEAD_DIM}-dim head, got {len(head_joint)}")
        self._send_request("request_moveh", {"joint": head_joint, "time": move_time})
        self.logger.debug("move_head sent: %s", head_joint)

    def _joint_pose_needs_second_joint(self, current: List[float]) -> bool:
        # 判断"当前是否太接近初始位/奇异构型"：如果左右臂的前几个关节角都接近 0，
        # 说明机器人可能"贴着"初始位附近，此时直接 movej 到 init 可能走一条
        # 奇怪的超短路径（机器人侧路径规划器会选出危险轨迹），需要先绕中间位。
        return (
            abs(current[0]) < 0.1
            and abs(current[8]) < 0.1
            and abs(current[3]) < 0.2
            and abs(current[11]) < 0.2
        )

    def _ee_pose_needs_second_joint(self, ee_pose: Dict) -> bool:
        # 另一种绕行触发条件：若当前末端位置太低（Z 低于配置阈值 init_ee_z_min），
        # 直接走 movej 可能撞地/干涉，也要先绕中间位。
        threshold = self.config.init_ee_z_min
        if threshold is None:
            return False

        z_values = []
        for key in ("left_position", "right_position"):
            position = ee_pose.get(key, [])
            if len(position) >= 3:
                try:
                    z_values.append(float(position[2]))   # 取 z 分量（下标 2）
                except (TypeError, ValueError):
                    continue
        if not z_values:
            raise StateError("bad ee pose: missing left/right z position")

        min_z = min(z_values)
        if min_z < float(threshold):
            self.logger.info(
                "ee z %.4f below init threshold %.4f; routing via second joint",
                min_z,
                threshold,
            )
            return True
        return False

    def _move_to_init_pose(self) -> None:
        """Bring the robot to the init pose declared in config.

        把机器人带到配置里声明的初始位姿，供 env.reset 等场景调用。
        流程：
          1. 若配置了 init_joints：先读当前状态；
          2. 判断是否需要"绕中间位"（贴近初始位 或 末端太低）；
             需要就先 movej 到 _second_joint（一个预置的安全中间构型）；
          3. movej 到真正的 init_joints，睡 2 秒等它走完；
          4. 若配置了 init_head：move_head 到指定头部位姿；
          5. 夹爪张开到 100（全开），再睡 2 秒。
        每一步之间用 time.sleep 等待机器人实际完成运动，是"阻塞式初始化"。
        """
        self.logger.info("moving to init pose")
        if self.config.init_joints is not None:
            states = self.get_joint_state(timeout=1.0)
            current = states["states"]
            # If lying close to home, swing through a known intermediate first to
            # avoid the controller picking a weird shortest path.
            need_second_joint = self._joint_pose_needs_second_joint(current)
            if not need_second_joint and self.config.init_ee_z_min is not None:
                need_second_joint = self._ee_pose_needs_second_joint(self.get_ee_poses(timeout=1.0))
            if need_second_joint:
                self.movej(self._second_joint, move_time=2.0)
                time.sleep(2)
            self.movej(self.config.init_joints, move_time=2.0)
            time.sleep(2)  # 2.1.5 controller bug: movej state is clobbered by move_head
        if self.config.init_head is not None:
            self.move_head(self.config.init_head, move_time=1.0)
            time.sleep(0.1)
        self.set_gripper(left_opening=100, right_opening=100)
        time.sleep(2)
        self.logger.info("init pose reached")
