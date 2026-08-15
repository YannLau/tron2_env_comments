"""Bridge WebSocket 观测提供者 —— TRON2 机器人的"眼睛和本体感觉"来源。

================================ 一句话定位 ================================

通过算力模块上的 ROS bridge WebSocket，订阅**图像**和**关节状态**话题；
后台线程内部跑 asyncio 事件循环收流，把图像与关节做**时间对齐**后，
经线程安全队列交给主线程使用。**控制命令不经过本模块**——动作仍由
Tron2 WebSocket 直连机器人下发（见 transport/websocket.py）。

============================== 为什么要 bridge ==============================

机器人自带的直连 WebSocket 只能回传关节状态，没有图像；相机图像由
算力模块采集，并发布到 ROS 话题上。bridge 就是两者之间的"翻译官"：
它替我们把 ROS 话题里的图像/关节数据以 WebSocket 流推出来，本模块负责
订阅、解析、对齐，最后拼成 env.py 需要的标准观测格式（18 维 state）。

=============================== 整体数据流 ===============================

   算力模块上的 ROS bridge 服务
        │   wss://<host>/bridge/ws?topic=...&kind=image|joint
        ▼
   BridgeObservationProvider.start()
        │   启动一个后台 daemon 线程
        ▼
   _run_loop()  →  asyncio.run(_async_main())
        │   为每个图像/关节话题各起一个订阅 task，
        │   另起一个 _align_dispatcher task 负责对齐分发
        ▼
   订阅 task 收帧：
        ├─ 图像 → 二进制 BRDG 帧 → parse_brdg_frame() → decode_image()
        │        → ImageFrame → aligner.push_image()
        └─ 关节 → JSON 文本帧    → parse_joint_message() → JointFrame
                 → aligner.push_joint()
        ▼
   TopicAligner.try_align()（ApproximateTime 近似时间对齐）
        │   以最早的图像时间戳为参考，为每个关节话题匹配时间最接近的帧
        ▼
   build_openpi_observation()（拼 18 维 state + OpenPI 命名的图像）
        │   放入 queue.Queue（线程安全、容量 10）
        ▼
   主线程调用 get_obs() 拿到最新观测（自动丢弃队列里更旧的）

================================ 快速上手 ================================

    from tron2_env.bridge import BridgeConfig, BridgeObservationProvider

    # 1. 把 BRIDGE_HOST 换成算力模块实际地址（默认值只是占位符）
    config = BridgeConfig(host="wss://192.168.1.20")
    provider = BridgeObservationProvider(config)

    # 2. 启动后台订阅线程，阻塞取观测（内部已对齐）
    provider.start()
    obs = provider.get_obs(timeout=1.0)
    # obs["images"]: {"cam_high": ..., "cam_left_wrist": ..., ...}  (RGB ndarray)
    # obs["state"]:  np.ndarray(18,)  [左臂7, 左夹爪1, 右臂7, 右夹爪1, 头2]

    # 3. 用完记得停止（也支持 with 语法：with provider: ...）
    provider.stop()

更常见的用法是不直接碰本模块，而是在 env.py 里设
``EnvConfig(observation_source="bridge", bridge_config=...)`` 一键启用。
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import queue
import ssl
import struct
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


# ============================================================================
# 默认话题配置
# ============================================================================

# 注意：以下 HOST 默认值都是**占位符**，不改成算力模块的真实地址无法使用。
DEFAULT_HOST = "wss://BRIDGE_HOST"          # WebSocket 服务地址（wss = 加密 WebSocket）
DEFAULT_HTTP_HOST = "https://BRIDGE_HOST"   # 对应的 HTTP 地址（调试/健康检查等场景备用）
DEFAULT_WS_PATH = "/bridge/ws"              # WebSocket 订阅入口路径

# 默认图像话题：key 是代码里使用的**逻辑名**，value 是算力模块上的 **ROS 话题名**。
# 三个相机：左右腕部（wrist）相机 + 顶部（top）全局相机。
# 带 "compressed" 的是压缩图像话题（带宽低、延迟小，通常选它）。
DEFAULT_IMAGE_TOPICS = {
    "camera_left": "/camera/left/color/image_resized/compressed",
    "camera_right": "/camera/right/color/image_resized/compressed",
    "camera_top": "/camera/top/color/image_raw/compressed",
}
# 默认关节话题：
#   * /joint_states     —— 16 维 [左臂7, 右臂7, 头部2]（不含夹爪）
#   * /gripper_state    —— 2 维 [左夹爪, 右夹爪]，开度 0-100
DEFAULT_JOINT_TOPICS = {
    "joint_states": "/joint_states",
    "gripper": "/gripper_state",
}


# ============================================================================
# Bridge 配置
# ============================================================================

@dataclass
class BridgeConfig:
    """Bridge WebSocket 观测配置（所有字段都有默认值，按需覆盖）。

    被 env.py 的 ``EnvConfig.bridge_config`` 持有，在
    ``observation_source="bridge"`` 时生效；也可以独立实例化，
    配合 :class:`BridgeObservationProvider` 单独使用。
    """

    # bridge 服务地址，形如 "wss://192.168.1.20"。
    # 默认值 "wss://BRIDGE_HOST" 只是占位符，必须换成算力模块真实地址。
    # 注意要带 wss:// 前缀（TLS 加密连接）。
    host: str = DEFAULT_HOST

    # WebSocket 订阅路径，拼接在 host 之后。一般无需修改。
    ws_path: str = DEFAULT_WS_PATH

    # 图像话题的限频（帧/秒）。0 表示不限频，按 bridge 侧的默认频率推送。
    # 每个订阅任务的 URL 里会带上 max_fps 参数（见 ws_url()）。
    image_max_fps: int = 0

    # 时间对齐允许的最大延迟（毫秒）：关节帧与参考图像时间戳之差超过该值时，
    # 本次对齐放弃，等待更新的数据（见 TopicAligner）。200ms 是经验值。
    align_max_delay_ms: int = 200

    # 是否校验 TLS 证书。bridge 通常用自签名证书，所以默认 False（跳过校验）。
    # 若 bridge 配置了正式证书，可设为 True 更安全。
    verify_tls: bool = False

    # 图像话题映射 {逻辑名: ROS 话题名}。订阅什么相机由这里决定。
    # 注意用 field(default_factory=...) 而不是直接传 dict：避免同一个
    # 可变默认值对象被所有实例共享（dataclass 的经典坑）。
    image_topics: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_IMAGE_TOPICS))

    # 关节话题映射 {逻辑名: ROS 话题名}。
    # 特殊用法：设成 {} 可完全关闭关节订阅（env.py 的
    # bridge_state_source="legacy" 模式会这么做：图像走 bridge，
    # state 改从机器人直连 WebSocket 现拉）。
    joint_topics: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_JOINT_TOPICS))

    # 是否把每步观测的图像存到本地磁盘（调试用，消耗磁盘 IO）。
    # 实际存图逻辑在 env.py 的 _save_debug_images_bridge()。
    save_debug_images: bool = True

    # 调试图像的保存目录（save_debug_images=True 时生效）。
    debug_image_dir: str = "./debug_images"


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BrdgFrame:
    """一条 BRDG 二进制帧的解析结果（bridge 私有协议的外层信封）。

    BRDG 是算力模块自定义的二进制封装格式，图像等大数据都以它打包推送。
    字段由 parse_brdg_frame() 从原始字节里解出。
    """
    mime: str            # MIME 类型，如 "image/jpeg" / "application/x-ros-image"
    timestamp_ms: int    # 帧时间戳（毫秒，Unix 毫秒）
    payload: bytes       # 实际载荷（对图像而言是压缩字节流或 RIMG 原始图像）


@dataclass
class ImageFrame:
    """一帧已解码的图像 + 其来源信息（对齐器消费的数据单元）。"""
    key: str             # 逻辑名（如 "camera_left"），对应 BridgeConfig.image_topics 的 key
    topic: str           # ROS 话题名
    timestamp_ms: int    # 帧时间戳（毫秒）
    image: np.ndarray    # 解码后的 RGB 图像，uint8，形状 (H, W, 3)


@dataclass
class JointFrame:
    """一帧关节状态 + 其来源信息（对齐器消费的数据单元）。

    positions / velocities / efforts 与 ROS sensor_msgs/JointState
    的字段一一对应，names 是关节名列表（同一话题内顺序固定）。
    """
    key: str             # 逻辑名（如 "joint_states"），对应 BridgeConfig.joint_topics 的 key
    topic: str           # ROS 话题名
    timestamp_ms: int    # 帧时间戳（毫秒）
    names: list[str]
    positions: list[float]    # 各关节位置（臂为弧度，夹爪话题为 0-100 开度）
    velocities: list[float]
    efforts: list[float]


# ============================================================================
# 协议解析
# ============================================================================

def make_ssl_ctx(insecure: bool = True) -> ssl.SSLContext:
    """构建 SSL 上下文。

    bridge 通常使用**自签名证书**（没有 CA 背书），默认 ``insecure=True``：
    关闭主机名校验和证书校验（CERT_NONE），否则握手会直接失败。
    如果 bridge 换成了正式证书，可传 ``insecure=False`` 走标准校验。
    """
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def parse_brdg_frame(data: bytes) -> Optional[BrdgFrame]:
    """解析 BRDG 二进制 WebSocket 消息为 :class:`BrdgFrame`。

    BRDG 帧格式（全部小端序）：:

        ┌─────────┬─────────┬──────────┬──────────────┬──────────────┐
        │ "BRDG"  │ version │ mime_len │ mime 字符串   │ timestamp_ms │  payload
        │  4 字节 │  1 字节 │  1 字节  │ mime_len 字节 │  8 字节 uint │  剩余字节
        └─────────┴─────────┴──────────┴──────────────┴──────────────┘

    因此最小合法帧长 = 4 + 1 + 1 + 8 = 14 字节。

    返回 ``None`` 表示**无法解析**（长度不足 / 魔数不对 / 版本不支持），
    调用方直接跳过这帧即可——流里可能混有其它协议的消息。
    """
    if len(data) < 14 or data[:4] != b"BRDG":
        return None
    version = data[4]
    if version != 1:
        return None
    mime_len = data[5]
    mime_start = 6
    mime_end = mime_start + mime_len
    ts_end = mime_end + 8
    if len(data) < ts_end:
        return None
    mime = data[mime_start:mime_end].decode("utf-8", errors="replace")
    timestamp_ms = struct.unpack_from("<Q", data, mime_end)[0]
    return BrdgFrame(mime=mime, timestamp_ms=timestamp_ms, payload=data[ts_end:])


def decode_image(frame: BrdgFrame) -> Optional[np.ndarray]:
    """把 BRDG 帧解码成 RGB uint8 图像数组，形状 (H, W, 3)。

    支持两种载荷：

    1. **标准压缩格式**（mime 为 image/jpeg / image/png）：
       直接用 PIL 解码并统一转成 RGB。

    2. **RIMG 原始图像**（mime 为 application/x-ros-image）：
       bridge 自定义的裸像素封装，省去编解码开销，格式为::

           ┌────────┬───────┬────────┬──────┬──────────┬───────────────┬────────────┐
           │ "RIMG" │ width │ height │ step │ 0(保留)  │ enc_len       │ encoding   │ raw
           │ 4 字节 │ u32   │ u32    │ u32  │ 1 字节   │ 1 字节        │ 变长字符串 │ 像素数据
           └────────┴───────┴────────┴──────┴──────────┴───────────────┴────────────┘

       其中：
         * ``step`` —— 每行字节数（行跨度）。因内存对齐可能大于
           ``width * 通道数``，多余部分是行尾 padding，解码时要裁掉；
         * encoding 支持 ``rgb8`` / ``bgr8``（3 通道）和 ``mono8``（单通道灰度）。
           bgr8 会翻转通道变 RGB，mono8 会复制成 3 通道 RGB。

    返回 ``None`` 表示该帧不是图像或数据损坏，调用方跳过即可。
    """
    if frame.mime in ("image/jpeg", "image/png"):
        image = Image.open(io.BytesIO(frame.payload)).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    if frame.mime != "application/x-ros-image":
        return None

    payload = frame.payload
    # RIMG 头最小长度：4 魔数 + 3*4 字段 + 1 保留 + 1 编码长度 = 18 字节
    if len(payload) < 18 or payload[:4] != b"RIMG":
        return None

    # 依次取出 32 位小端无符号整数：宽 / 高 / 行跨度
    width = struct.unpack_from("<I", payload, 4)[0]
    height = struct.unpack_from("<I", payload, 8)[0]
    step = struct.unpack_from("<I", payload, 12)[0]
    # 字节 16 是保留字节（值为 0），字节 17 是 encoding 字符串长度
    enc_len = payload[17]
    enc_start = 18
    enc_end = enc_start + enc_len
    if len(payload) < enc_end:
        return None

    encoding = payload[enc_start:enc_end].decode("utf-8", errors="replace")
    raw = payload[enc_end:]
    if step <= 0:
        return None

    # --- 3 通道：rgb8 / bgr8 ---
    if encoding in ("rgb8", "bgr8"):
        expected = step * height
        if len(raw) < expected:
            return None
        # 按 step 为行跨度重塑，再裁掉每行末尾的 padding 字节和多余列
        rows = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, step)
        arr = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]   # 通道翻转：BGR → RGB
        return arr.copy()

    # --- 单通道灰度：mono8 ---
    if encoding == "mono8":
        expected = step * height
        if len(raw) < expected:
            return None
        rows = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, step)
        arr = rows[:, :width]
        return np.stack([arr, arr, arr], axis=-1).copy()   # 复制成 3 通道，冒充 RGB

    return None


def parse_joint_message(raw: str, key: str, default_topic: str) -> Optional[JointFrame]:
    """解析 bridge 的 JSON 文本帧为 :class:`JointFrame`。

    bridge 的 WebSocket 文本通道不只推关节数据（还有状态/心跳等消息），
    因此先看 ``type`` 字段：只有 ``"joint_state"`` 才处理，其余一律忽略。

    时间戳缺失时回退到本机当前时间（``time.time()*1000``），保证对齐器
    至少有可比对的数值。
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if msg.get("type") != "joint_state":
        return None

    data = msg.get("data", {})
    return JointFrame(
        key=key,
        topic=msg.get("topic", default_topic),
        timestamp_ms=int(msg.get("timestamp", int(time.time() * 1000))),
        names=list(data.get("names", [])),
        positions=list(data.get("positions", [])),
        velocities=list(data.get("velocities", [])),
        efforts=list(data.get("efforts", [])),
    )


def ws_url(host: str, path: str, topic: str, kind: str, max_fps: Optional[int] = None) -> str:
    """构造 bridge WebSocket 订阅 URL。

    形如::

        wss://<host>/bridge/ws?topic=/camera/left/...&kind=image&max_fps=15

    参数含义：
      * ``topic``   —— 要订阅的 ROS 话题名；
      * ``kind``    —— 通道类型："image"（推二进制 BRDG 帧）或 "joint"
                      （推 JSON 文本帧），服务端据此决定编码方式；
      * ``max_fps`` —— 可选限频，None 表示不限制。
    """
    query_params = {"topic": topic, "kind": kind}
    if max_fps is not None:
        query_params["max_fps"] = str(max_fps)
    return f"{host.rstrip('/')}{path}?{urllib.parse.urlencode(query_params)}"


# ============================================================================
# 时间对齐器
# ============================================================================

class TopicAligner:
    """ApproximateTime 近似时间对齐器：图像与关节的"对表"组件。

    ============================== 要解决什么问题 ==============================

    图像走压缩/编码链路、关节走文本链路，两路的延迟不同，到达时间天生错开。
    但下游 policy 需要**同一时刻**的图像 + 关节状态才能做决策。
    本类实现 ROS 里的 ApproximateTime 策略：不求精确同帧，只求"时间最接近"。

    ================================= 对齐算法 =================================

    1. **参考时间**：取所有已到达图像中**最早**的时间戳作 ref_ts
       —— 保证 ref_ts 时刻所有相机的图像都已就绪；
    2. **关节匹配**：对每个关节话题，在缓冲里选时间戳离 ref_ts 最近的帧；
       若时间差超过 max_delay_ms，说明数据对不上表，本次放弃，等更新数据；
    3. **防重复**：ref_ts 没前进（相机没来新帧）就绝不重复产出观测；
    4. **清旧**：匹配成功后把早于 ref_ts - max_delay_ms 的关节旧帧清掉，
       只保留比当前参考"更新或相差不大"的帧，缓冲永远有界。
    """

    def __init__(
        self,
        image_keys: Iterable[str],
        joint_keys: Iterable[str],
        max_delay_ms: int = 200,
        joint_buffer_size: int = 100,
    ):
        self.image_keys = list(image_keys)          # 需要对齐的全部图像逻辑名
        self.joint_keys = list(joint_keys)          # 需要对齐的全部关节逻辑名
        self.max_delay_ms = int(max_delay_ms)       # 关节 vs 参考图像允许的最大时间差
        # 每路图像只留"最新一帧"——对齐永远以最新状态为准
        self._latest_image: Dict[str, ImageFrame] = {}
        # 每路关节留一个定长缓冲（旧满丢旧），供"找时间最接近的帧"检索
        self._joint_buffer: Dict[str, collections.deque[JointFrame]] = {
            key: collections.deque(maxlen=joint_buffer_size) for key in self.joint_keys
        }
        # 上次成功产出的参考时间戳：ref_ts 不变就不重复产出
        self._last_emitted_ref_ts: Optional[int] = None
        # 上次成功产出的墙钟时间：用于诊断"多久没产出观测了"
        self._last_emit_wall_s: float = time.time()
        # stall 日志节流：距上次打印不足 1 秒就不再打印
        self._last_stall_log_s: float = 0.0
        # 连续 500ms 没产出观测才认为"卡住"，值得告警
        self._stall_warn_ms: float = 500.0

    def _log_stall(self, reason: str) -> None:
        """对齐器长时间没产出时的节流诊断日志。

        只在对齐停滞超过 _stall_warn_ms 后才记录，且 1 秒最多打印一次，
        避免高频刷屏（例如某路相机一直断线时）。
        """
        now = time.time()
        stalled_ms = (now - self._last_emit_wall_s) * 1000.0
        if stalled_ms < self._stall_warn_ms:
            return
        if now - self._last_stall_log_s >= 1.0:
            self._last_stall_log_s = now
            logger.warning("[bridge:align] no aligned observation for %.0fms: %s", stalled_ms, reason)

    def push_image(self, frame: ImageFrame):
        """图像订阅 task 调用：存入最新图像（同名 key 覆盖旧帧）。"""
        self._latest_image[frame.key] = frame

    def push_joint(self, frame: JointFrame):
        """关节订阅 task 调用：追加到对应话题的缓冲（满队自动丢最旧）。"""
        if frame.key in self._joint_buffer:
            self._joint_buffer[frame.key].append(frame)

    def try_align(self) -> Optional[Dict[str, Any]]:
        """尝试做一次对齐，成功返回观测字典，失败返回 None（继续等数据）。

        由 _align_dispatcher 每 1ms 调用一次。返回的字典结构::

            {
                "images":       {逻辑名: np.ndarray},   # 各相机最新 RGB 图像
                "image_frames": {逻辑名: ImageFrame},    # 带时间戳的完整帧
                "joint_states": {逻辑名: JointFrame},    # 与 ref_ts 匹配的关节帧
                "timestamp_ms": ref_ts,                  # 本次对齐的参考时间
            }
        """
        # 第 1 步：所有相机都至少来过一帧，才谈得上对齐
        missing = [key for key in self.image_keys if key not in self._latest_image]
        if missing:
            self._log_stall("waiting for first image frame: " + ", ".join(missing))
            return None

        # 第 2 步：参考时间 = 最早图像时间戳（保证所有图像都不晚于它）
        img_ts = {key: self._latest_image[key].timestamp_ms for key in self.image_keys}
        ref_ts = min(img_ts.values())
        # 参考时间没前进 → 没有新图像 → 不重复产出旧观测
        if self._last_emitted_ref_ts == ref_ts:
            newest = max(img_ts.values())
            lagging = ", ".join(
                f"{key} lags {newest - ts}ms" for key, ts in img_ts.items() if newest - ts > 0
            ) or "image timestamps are equal"
            self._log_stall(f"image reference timestamp has not advanced: ref_ts={ref_ts}ms; {lagging}")
            return None

        # 第 3 步：每个关节话题在缓冲里挑时间最接近 ref_ts 的帧，
        #         时间差超限则本次放弃（数据还没对齐，宁缺毋滥）
        matched: Dict[str, JointFrame] = {}
        for key in self.joint_keys:
            buffer = self._joint_buffer[key]
            if not buffer:
                self._log_stall(f"joint topic {key} has no buffered frames")
                return None
            best = min(buffer, key=lambda f: abs(f.timestamp_ms - ref_ts))
            dt = abs(best.timestamp_ms - ref_ts)
            if dt > self.max_delay_ms:
                self._log_stall(
                    f"joint topic {key} is {dt}ms from image ref {ref_ts}ms "
                    f"(limit {self.max_delay_ms}ms, buffer={len(buffer)}, best={best.timestamp_ms}ms)"
                )
                return None
            matched[key] = best

        # 第 4 步：清理过期的关节数据——比 ref_ts 还旧 max_delay_ms 以上的帧
        #         以后再也匹配不上了（参考时间只会前进不会后退）。
        #         条件 len(buffer) > 1 保证每个话题至少保留一帧，防止
        #         清理过度导致下一步"no buffered frames"。
        for key, buffer in self._joint_buffer.items():
            while len(buffer) > 1 and buffer[0].timestamp_ms < ref_ts - self.max_delay_ms:
                buffer.popleft()

        # 第 5 步：记录产出状态，返回对齐结果
        self._last_emitted_ref_ts = ref_ts
        self._last_emit_wall_s = time.time()
        return {
            "images": {key: self._latest_image[key].image for key in self.image_keys},
            "image_frames": dict(self._latest_image),
            "joint_states": matched,
            "timestamp_ms": ref_ts,
        }


# ============================================================================
# 观测格式转换
# ============================================================================

def build_openpi_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """把 bridge 对齐全的观测转换为 TRON2/OpenPI 惯例格式。

    ============================== 维度拼接规则 ==============================

    TRON2 惯例（state 18 维）: [左臂7, 左夹爪1, 右臂7, 右夹爪1, 头部2]
    Bridge /joint_states  16 维: [左臂7, 右臂7, 头部2]          （不含夹爪）
    Bridge /gripper_state  2 维: [左夹爪, 右夹爪]，开度 0-100

    拼接时按 TRON2 顺序把夹爪插进两臂之间；夹爪 0-100 归一化到 0-1。
    若任一话题维度不足（例如订阅被关闭、消息异常），退化为"直接拼接"，
    宁可用残缺数据也不崩溃，由下游自行判断。

    ============================== 图像命名惯例 ==============================

    bridge 逻辑名（camera_top / camera_left / camera_right）映射为
    OpenPI 风格名（cam_high / cam_left_wrist / cam_right_wrist）；
    某相机没数据时该项直接剔除，不占位。

    ============================== 返回结构 ==============================

        {
            "images":   {"cam_high": ndarray, "cam_left_wrist": ndarray, ...},
            "state":    np.ndarray(18,)  dtype=float32,
            "metadata": {...}  # 各种时间戳，便于下游排查延迟
        }
    """
    images = obs["images"]
    joints = obs["joint_states"].get("joint_states")   # 16 维关节帧（可能为 None）
    gripper = obs["joint_states"].get("gripper")       # 2 维夹爪帧（可能为 None）

    # bridge 逻辑名 → OpenPI 惯例名
    openpi_images = {
        "cam_high": images.get("camera_top"),
        "cam_left_wrist": images.get("camera_left"),
        "cam_right_wrist": images.get("camera_right"),
    }
    openpi_images = {k: v for k, v in openpi_images.items() if v is not None}

    # 对应每个 OpenPI 名保留完整帧信息（含时间戳）
    image_frame_map = {
        "cam_high": obs.get("image_frames", {}).get("camera_top"),
        "cam_left_wrist": obs.get("image_frames", {}).get("camera_left"),
        "cam_right_wrist": obs.get("image_frames", {}).get("camera_right"),
    }
    image_timestamps_ms = {
        key: frame.timestamp_ms
        for key, frame in image_frame_map.items()
        if frame is not None
    }

    joint_pos = np.asarray(joints.positions if joints else [], dtype=np.float32)
    gripper_pos = np.asarray(gripper.positions if gripper else [], dtype=np.float32) / 100.0

    # 主路径：两个话题都齐全 → 按 TRON2 18 维惯例精确拼接
    if joint_pos.shape[0] >= 16 and gripper_pos.shape[0] >= 2:
        state = np.concatenate([
            joint_pos[:7],       # 左臂
            gripper_pos[:1],     # 左夹爪
            joint_pos[7:14],     # 右臂
            gripper_pos[1:2],    # 右夹爪
            joint_pos[14:16],    # 头部
        ])
    # 退化路径：维度不足时直接拼接（防御性处理，宁可残缺不崩溃）
    else:
        state = np.concatenate([joint_pos, gripper_pos])

    bridge_ref_timestamp_ms = obs.get("timestamp_ms")
    image_timestamp_ms = (
        min(image_timestamps_ms.values()) if image_timestamps_ms else bridge_ref_timestamp_ms
    )
    metadata = {
        "observation_source": "bridge",
        "bridge_ref_timestamp_ms": bridge_ref_timestamp_ms, # 本次 observation 以哪个图像时间为对齐基准
        "joint_timestamp_ms": joints.timestamp_ms if joints else None, # 被对齐的关节状态时间
        "gripper_timestamp_ms": gripper.timestamp_ms if gripper else None,
        "image_timestamp_ms": image_timestamp_ms, #当前图像组的最早图像时间，基本等于 ref
        "image_timestamps_ms": image_timestamps_ms, #每个相机各自是什么时间的图像
    }

    return {
        "images": openpi_images,
        "state": state.astype(np.float32, copy=False),
        "metadata": metadata,
    }


# ============================================================================
# Bridge 观测提供者
# ============================================================================

class BridgeObservationProvider:
    """后台线程从 bridge WebSocket 获取对齐全的观测数据。

    ============================== 线程模型 ==============================

    对外是普通的阻塞接口（start / get_obs / stop），内部则是"线程 + 事件循环"：

      * **后台线程**：start() 创建一个 daemon 线程，其入口 _run_loop()
        调用 ``asyncio.run(_async_main())``，在该线程里跑整套订阅逻辑；
      * **桥接队列**：_obs_queue 是线程安全的 queue.Queue（容量 10），
        后台线程把对齐好的观测放进去，主线程 get_obs() 取出来——
        这是 asyncio 世界与同步世界之间唯一的通道。

    ============================== 生命周期 ==============================

      start() → 后台线程 + N 个订阅 task + 1 个对齐分发 task
      get_obs(timeout) → 阻塞取最新观测（自动丢掉队列里更旧的）
      stop() → 置停止标志、取消所有 task、join 线程（最多等 5 秒）
      也支持 ``with provider:`` 语法自动 start/stop。

    控制命令不经过此类，仍由 Tron2 WebSocket 直连机器人。
    """

    def __init__(self, config: BridgeConfig):
        self.config = config
        # 桥接队列：后台 asyncio 线程 → 主线程。maxsize=10 起到
        # "天然限流"作用——主线程消费慢了就丢弃旧观测（见 _align_dispatcher）。
        self._obs_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=10)
        # 停止标志：跨线程共享（threading.Event 是线程安全的），
        # 后台线程各处循环都在检查它。
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """启动后台订阅线程（幂等：重复调用无副作用）。"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        # daemon=True：主程序退出时后台线程不阻塞进程退出
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Bridge 观测提供者已启动 (host=%s, fps=%d)",
                     self.config.host, self.config.image_max_fps)

    def stop(self):
        """停止后台订阅线程（幂等：未启动时调用无副作用）。

        流程：置停止标志 → 后台循环退出 → 取消所有 task → join 线程
        （最多等 5 秒，超时就放弃，因为它是 daemon 线程不会卡住进程）。
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Bridge 观测提供者已停止")

    def get_obs(self, timeout: float = 1.0) -> Dict[str, Any]:
        """阻塞获取对齐全的观测数据（主线程唯一入口）。

        Args:
            timeout: 超时秒数

        Returns:
            最新观测字典::

                {
                    "images":  {"cam_high": ndarray(H,W,3), ...},
                    "state":   np.ndarray(18,)  float32,
                    "metadata": {...},
                }

        Raises:
            TimeoutError: 超时未获取到观测（常见原因：bridge 没连上、
                相机没连算力模块、或话题名配置错误）
        """
        try:
            obs = self._obs_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"Bridge 观测获取超时 ({timeout}s),请检查相机是否连接算力模块")

        # 重要细节：队列里可能已经积压了多条观测（policy 推理远比订阅慢）。
        # 依次把队列"排空"，只返回**最新**一条，保证推理用的是最新鲜的帧，
        # 而不是几秒前积压的旧数据。
        while True:
            try:
                obs = self._obs_queue.get_nowait()
            except queue.Empty:
                return obs

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------

    def _run_loop(self):
        """后台线程入口：运行 asyncio 事件循环。

        线程 body 只做一件事——``asyncio.run(_async_main())``。
        异常兜底：任何未捕获异常都会导致事件循环退出，这里记录错误，
        避免 daemon 线程静默死亡（之后 get_obs 会以超时形式暴露问题）。
        """
        try:
            asyncio.run(self._async_main())
        except Exception as e:
            logger.error("Bridge 事件循环异常退出: %s", e)

    async def _async_main(self):
        """异步主逻辑：订阅所有话题，等待停止信号。

        结构：
          1. 构建 SSL 上下文（自签名证书 → 跳过校验）；
          2. 创建 TopicAligner（图像 + 关节共用一个对齐器）；
          3. 每个图像话题起一个 _subscribe_image task，
             每个关节话题起一个 _subscribe_joint task，
             另起一个 _align_dispatcher task 做对齐分发；
          4. 主循环轮询停止标志，同时检查各 task 是否异常退出。
        """
        cfg = self.config
        ssl_ctx = make_ssl_ctx(insecure=not cfg.verify_tls)
        aligner = TopicAligner(
            image_keys=cfg.image_topics.keys(),
            joint_keys=cfg.joint_topics.keys(),
            max_delay_ms=cfg.align_max_delay_ms,
        )

        tasks = []
        for key, topic in cfg.image_topics.items():
            tasks.append(asyncio.create_task(
                self._subscribe_image(key, topic, aligner, ssl_ctx)
            ))
        for key, topic in cfg.joint_topics.items():
            tasks.append(asyncio.create_task(
                self._subscribe_joint(key, topic, aligner, ssl_ctx)
            ))
        # 对齐分发任务
        tasks.append(asyncio.create_task(
            self._align_dispatcher(aligner)
        ))

        try:
            # threading.Event 不能被 await（它不是 asyncio 原语），
            # 所以只能轮询。同时巡检各 task：订阅 task 内部有重连逻辑，
            # 正常不会结束——若真的结束且带着异常，说明某路订阅死了，
            # 提前打日志暴露出来，而不是伪装成"对齐静默卡住"。
            while not self._stop_event.is_set():
                for task in tasks:
                    if task.done():
                        exc = task.exception()
                        if exc is not None:
                            logger.error("[bridge] background task %s exited: %r", task.get_name(), exc)
                await asyncio.sleep(0.1)
        finally:
            # 停止时取消所有 task 并等待它们收尾（return_exceptions=True
            # 吞掉 CancelledError，避免 gather 把取消异常再抛出来）
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _subscribe_image(
        self, key: str, topic: str, aligner: TopicAligner, ssl_ctx: ssl.SSLContext
    ):
        """订阅图像话题（每话题一个 task，断线自动重连）。

        流程：连接 → 逐帧收 → 二进制 BRDG 帧解析 → 解码成 RGB → 推给对齐器。
        无限重连循环：任何异常（断网、服务重启等）都等 3 秒再连，
        保证 bridge 恢复后能自动续上。
        """
        import websockets

        cfg = self.config
        url = ws_url(cfg.host, cfg.ws_path, topic, "image", cfg.image_max_fps)
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ssl=ssl_ctx) as socket:
                    async for msg in socket:
                        if self._stop_event.is_set():
                            return
                        # 图像通道只收二进制帧；文本帧（状态/心跳）直接跳过
                        if isinstance(msg, str):
                            continue
                        if not isinstance(msg, bytes):
                            continue

                        frame = parse_brdg_frame(msg)
                        if frame is None:
                            continue
                        image = decode_image(frame)
                        if image is None:
                            continue

                        aligner.push_image(ImageFrame(key, topic, frame.timestamp_ms, image))
            except asyncio.CancelledError:
                raise   # 取消必须向上传播（asyncio 惯例），不能被下面的兜底吞掉
            except Exception as exc:
                logger.warning("[bridge:image] %s 断线: %s, 3秒后重连", key, exc)
                await asyncio.sleep(3)

    async def _subscribe_joint(
        self, key: str, topic: str, aligner: TopicAligner, ssl_ctx: ssl.SSLContext
    ):
        """订阅关节状态话题（每话题一个 task，断线自动重连）。

        与图像订阅对称：关节通道收的是 JSON **文本帧**，解析成
        JointFrame 后推给对齐器。
        """
        import websockets

        cfg = self.config
        url = ws_url(cfg.host, cfg.ws_path, topic, "joint")
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ssl=ssl_ctx) as socket:
                    async for msg in socket:
                        if self._stop_event.is_set():
                            return
                        # 关节通道只收文本帧；偶发的二进制帧直接跳过
                        if not isinstance(msg, str):
                            continue
                        frame = parse_joint_message(msg, key, topic)
                        if frame is None:
                            continue
                        aligner.push_joint(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[bridge:joint] %s 断线: %s, 3秒后重连", key, exc)
                await asyncio.sleep(3)

    async def _align_dispatcher(self, aligner: TopicAligner):
        """定期轮询对齐器，把结果放入线程安全队列（每 1ms 一次）。

        队列满时丢弃最旧的观测、放入最新的——观测数据以"最新"为第一优先级，
        旧观测对实时控制没有价值。
        """
        while not self._stop_event.is_set():
            obs = aligner.try_align()
            if obs is not None:
                openpi_obs = build_openpi_observation(obs)
                try:
                    self._obs_queue.put_nowait(openpi_obs)
                except queue.Full:
                    # 队列满时丢弃最旧的观测
                    try:
                        self._obs_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._obs_queue.put_nowait(openpi_obs)
                    except queue.Full:
                        pass
            await asyncio.sleep(0.001)  # 1ms 轮询间隔

    # --- 上下文管理器协议：with BridgeObservationProvider(...) as p: ... ---
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# 测试辅助函数
# ============================================================================

def make_brdg(mime: str, timestamp_ms: int, payload: bytes) -> bytes:
    """构造 BRDG 二进制帧（测试用）。

    与 parse_brdg_frame() 互为逆操作：构造 → 解析 → 应还原出相同字段，
    便于单测里不依赖真实 bridge 就能验证解析逻辑。
    """
    mime_bytes = mime.encode("utf-8")
    return b"BRDG" + bytes([1, len(mime_bytes)]) + mime_bytes + struct.pack("<Q", timestamp_ms) + payload


def make_rimg(width: int, height: int, encoding: str, raw: bytes, step: Optional[int] = None) -> bytes:
    """构造 RIMG 原始图像帧（测试用）。

    与 decode_image() 的 RIMG 分支互为逆操作。``step`` 缺省时按
    "无 padding" 计算：3 通道编码为 width*3，单通道为 width。
    """
    enc = encoding.encode("utf-8")
    step = step if step is not None else width * (3 if encoding in ("rgb8", "bgr8") else 1)
    header = b"RIMG" + struct.pack("<III", width, height, step) + bytes([0, len(enc)]) + enc
    return header + raw
