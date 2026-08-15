"""RealSense 多相机采集管理器 —— legacy 观测路径的"眼睛"部分。

================================ 一句话定位 ================================

用 Intel RealSense 相机（D455 顶部全局相机 + D405 腕部相机），每台相机
一个独立采集线程，持续拉彩色帧放进有界队列，主线程按需取"最新一帧"。

================================ 与 bridge 的关系 ================================

env.py 的观测来源是二选一的（``EnvConfig.observation_source``）：

  * **legacy**（本模块）：图像由 RealSense **直连本机 USB** 采集
    （MultiCameraManager），关节状态由机器人 WebSocket 直连获取；
  * **bridge**：图像 + 关节都从算力模块的 ROS bridge WebSocket 订阅
    （见 bridge.py），本模块完全不用。

本模块代码里出现的 cam_high / cam_left_wrist / cam_right_wrist 命名，
与 bridge 模式的 cam_high / cam_left_wrist / cam_right_wrist 完全一致，
下游 policy 拿到的图像 key 相同——两种观测来源对上层透明。

================================ 线程模型 ================================

    MultiCameraManager
        │ start_capture()
        ▼
    ├─ Thread("cam-cam_high")       ──► pipeline.wait_for_frames()
    ├─ Thread("cam-cam_left_wrist") ──► pipeline.wait_for_frames()  每线程独立，
    └─ Thread("cam-cam_right_wrist")──► pipeline.wait_for_frames()  互不阻塞
              │ 各自 append 帧字典
              ▼
    frame_queues = {name: deque(maxlen=10)}   # 有界队列：慢消费丢旧帧
              │ 主线程 get_all_latest_frames() 取最新一帧
              ▼
    env.py（BGR → RGB 转换、分辨率校验、延迟统计）

要点：
  * 每个相机独立线程 + 独立 pipeline，避免串行 wait_for_frames 互相拖慢；
  * 帧队列 maxlen 有界：主线程消费慢时自动丢**最旧**的帧，永远保留最新；
  * 彩色流格式是 **BGR**（rs.format.bgr8），转 RGB 由 env.py 负责。

================================ 快速上手 ================================

    from tron2_env.camera import MultiCameraManager

    # 把占位符换成你手上相机的真实序列号（贴在相机机身标签上）
    serial_to_name = {
        "123456789012": "cam_high",
        "234567890123": "cam_left_wrist",
        "345678901234": "cam_right_wrist",
    }
    with MultiCameraManager(serial_to_name=serial_to_name) as cm:   # 自动 start/stop
        frames = cm.get_all_latest_frames()   # {"cam_high": {...}, ...}
        bgr = frames["cam_high"]["color"]     # np.ndarray (H, W, 3) uint8 BGR

也可直接跑本文件做冒烟测试（把各相机最新帧存成 PNG）：::

    python -m tron2_env.camera

============================== 依赖与注意事项 ===============================

  * 依赖 pyrealsense2（RealSense SDK 的 Python 绑定，不在核心依赖里，
    env.py 在需要时才 import，装不上就走 bridge 观测路径）；
  * 序列号映射里的默认值 "HEAD_CAMERA_SERIAL" 等都是**占位符**，
    不替换成真实序列号时，setup_pipelines 会把这些相机跳过；
  * 深度流默认关闭（代码里注释掉了），TRON2 任务只用彩色图像。
"""

import logging
import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread, Lock
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# Multi-Camera Manager
# ============================================================================

class MultiCameraManager:
    """RealSense 多相机管理器。

    每个相机在独立线程中采集，避免串行 ``wait_for_frames`` 阻塞
    （三个相机若按顺序等帧，60fps 会被拖成 20fps 都不到）。

    支持 D455（cam_high，顶部全局相机）和 D405（wrist，腕部相机）的
    高帧率模式（默认 640x480 @ 60fps）。

    线程模型细节见模块 docstring 的示意图。
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        serial_to_name: Optional[Dict[str, str]] = None,
        camera_configs: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """初始化多相机管理器。

        注意：本方法**只做配置与数据结构准备**，不检测也不启动相机。
        真正枚举设备发生在 start_capture() → setup_pipelines()。

        Args:
            max_queue_size: 每个相机帧队列的最大长度（帧）。主线程消费
                慢于采集时，队列满自动丢最旧帧，始终保留最新 10 帧。
            serial_to_name: 相机序列号 → 逻辑名 的映射，如
                {"123456789012": "cam_high"}。不传则用占位符默认值
                （必须替换成真实序列号才有用）。
            camera_configs: 每台相机的采集参数 {逻辑名: {color_width,
                color_height, fps}}。不传则全部用 640x480@60。
        """
        self._setup_logger()

        self.max_queue_size = max_queue_size
        self.running = False        # 采集开关：采集线程的 while 条件，跨线程共享
        self.lock = Lock()          # 保护 time_stamps 的读写（见 _camera_thread / get_timestamp_history）

        # 序列号 → 逻辑名 映射。键是贴在相机机身上的序列号（detect_cameras()
        # 返回的就是它），值是代码里使用的名字。默认值全是占位符！
        self.serial_to_name = serial_to_name or {
            "HEAD_CAMERA_SERIAL": "cam_high",
            "LEFT_WRIST_CAMERA_SERIAL": "cam_left_wrist",
            "RIGHT_WRIST_CAMERA_SERIAL": "cam_right_wrist",
        }

        # 每台相机的采集参数。注意 key 是**逻辑名**（与 serial_to_name 的
        # value 一致），不是序列号。空字典等价于不传 → 用默认配置。
        if camera_configs:
            self.camera_configs = camera_configs
        else:
            self.camera_configs = {
                'cam_left_wrist': {'color_width': 640, 'color_height': 480, 'fps': 60},
                'cam_right_wrist': {'color_width': 640, 'color_height': 480, 'fps': 60},
                'cam_high': {'color_width': 640, 'color_height': 480, 'fps': 60}
            }

        # 已启动的 pipeline，start_capture() → setup_pipelines() 里填充
        self.pipeline_dict = {}
        # 帧队列：{逻辑名: deque(maxlen=max_queue_size)}。生产者是采集线程
        # （append），消费者是主线程（pop）。deque 的 append/pop 在 CPython
        # 下是原子的（GIL 保证），所以这里不加锁。
        self.frame_queues = {name: deque(maxlen=max_queue_size) for name in self.camera_configs}
        # 时间戳历史：{逻辑名: deque(maxlen=100)}，给 env.py 做延迟统计用。
        # 与帧队列不同，这里读写加了 self.lock（历史记录会被遍历，
        # 遍历中途被 append 可能触发 RuntimeError）。
        self.time_stamps = {name: deque(maxlen=100) for name in self.camera_configs}

        # 采集线程表：{逻辑名: Thread}，stop_capture() 据此 join
        self._capture_threads: Dict[str, Thread] = {}

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        """从配置字典创建管理器实例（供 env.py 的 raw_config 路径使用）。

        读取 ``config_dict["camera"]`` 子字典，键如下::

            {
              "camera": {
                "serial_to_name": {"<序列号>": "cam_high", ...},  # 可选
                "resolution": [H, W],        # 可选，默认 [480, 640]
                "fps": 30,                   # 可选，默认 30
                "camera_names": [...],       # 可选，缺省时由 serial_to_name 推导
                "max_queue_size": 10,        # 可选
              }
            }

        注意 ``resolution`` 是 **[高, 宽]** 顺序（图像惯例），而内部
        color_height / color_width 字段按 RealSense 的 [宽, 高] 存储，
        转换发生在下面两行 —— 这是最容易看错的地方。
        """
        camera_cfg = config_dict.get('camera', {})

        # 提取序列号映射（如果存在）
        serial_to_name = camera_cfg.get('serial_to_name')

        # 构造相机配置 (resolution = [H, W], default 480x640)
        res = camera_cfg.get('resolution', [480, 640])
        fps = camera_cfg.get('fps', 30)

        # camera_names 可以显式指定，也可以从 serial_to_name 的 values 推导。
        # 两者都没有时列表为空 → camera_configs 为空 → 构造器里走默认配置。
        camera_names = camera_cfg.get('camera_names') or (
            list(serial_to_name.values()) if serial_to_name else []
        )
        camera_configs = {
            name: {'color_width': res[1], 'color_height': res[0], 'fps': fps}
            for name in camera_names
        }

        return cls(
            max_queue_size=camera_cfg.get('max_queue_size', 10),
            serial_to_name=serial_to_name,
            # 空字典 → None，让构造器 fallback 到默认三相机配置
            camera_configs=camera_configs if camera_configs else None
        )

    def _setup_logger(self):
        """初始化本模块专用 logger（CameraManager）。

        注意 propagate=False：日志只输出到自己的 StreamHandler，
        不再向上冒泡到 root logger，避免 env.py 等上层已配置好日志
        时出现重复打印。
        """
        self.logger = logging.getLogger("CameraManager")
        # 已注册过 handler 就复用（同一进程重复创建实例时避免重复输出）
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def detect_cameras(self) -> List[str]:
        """检测连接在本机的 RealSense 相机序列号。

        实现：用默认 context 查询全部 USB 设备，逐个读序列号。
        过滤掉序列号里带 "Asic" 的条目——RealSense SDK 在部分平台上
        会枚举出 ASIC 相关的非相机设备条目，它们不能开流、必须剔除。
        返回的序列号用于与 serial_to_name 匹配。
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        return [s for s in serials if 'Asic' not in s]

    def setup_pipelines(self):
        """配置所有相机的采集管道（只配置，不启动——启动在 start_capture）。

        流程：
          1. detect_cameras() 枚举本机相机；
          2. 逐个查 serial_to_name：序列号没有映射的相机打 warning 跳过
             （比如临时多插了一台相机，不影响已映射的相机）；
          3. 为每台相机创建独立 rs.pipeline() + rs.config()，
             按 camera_configs 的参数启用彩色流（bgr8 格式）。

        深度流默认关闭（下方注释掉的代码）：TRON2 任务只用彩色图像，
        开深度流徒增带宽和 CPU 开销；如需深度，取消注释即可。
        """
        serial_numbers = self.detect_cameras()
        self.logger.info(f"检测到 {len(serial_numbers)} 个相机: {serial_numbers}")

        if not serial_numbers:
            raise RuntimeError("未检测到 RealSense 相机")

        for serial in serial_numbers:
            if serial not in self.serial_to_name:
                self.logger.warning(f"序列号 {serial} 未定义映射名称，跳过")
                continue

            camera_name = self.serial_to_name[serial]
            # 未显式配置的相机兜底 640x480@30（通常来自默认 camera_configs）
            cam_cfg = self.camera_configs.get(camera_name, {'color_width': 640, 'color_height': 480, 'fps': 30})

            # 每台相机一个独立 pipeline + config，互不共享——
            # 这是"多相机并行采集"的关键，共享会导致设备争抢。
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)   # 把本 config 绑定到指定序列号的设备

            # 配置颜色流
            config.enable_stream(
                rs.stream.color,
                cam_cfg['color_width'], cam_cfg['color_height'],
                rs.format.bgr8, cam_cfg['fps']
            )

            # 配置深度流（默认关闭，见本方法 docstring）
            # config.enable_stream(
            #     rs.stream.depth,
            #     cam_cfg['color_width'], cam_cfg['color_height'],
            #     rs.format.z16, cam_cfg['fps']
            # )

            self.pipeline_dict[camera_name] = {
                'pipeline': pipeline,
                'config': config,
                'serial': serial
            }

    def start_capture(self):
        """开始采集——每个相机启动一个独立线程。

        幂等：已启动（running=True）时直接返回，重复调用无副作用。

        顺序：首次调用先 setup_pipelines()（枚举设备、建 pipeline），
        然后逐个 pipeline.start()（出错只记日志不中断，尽量让其它相机
        正常跑），最后为每个相机起 daemon 采集线程。
        """
        if self.running:
            return

        # 首次调用时才会真正去枚举设备、创建 pipeline
        if not self.pipeline_dict:
            self.setup_pipelines()

        for name, info in self.pipeline_dict.items():
            try:
                info['pipeline'].start(info['config'])
                self.logger.info(f"相机 {name} ({info['serial']}) 已启动")
            except Exception as e:
                # 单台相机启动失败不拖累全局：其余相机继续，出错的相机
                # 其采集线程会在 wait_for_frames 里反复失败（debug 日志）。
                self.logger.error(f"无法启动相机 {name}: {e}")

        self.running = True
        # 每个相机一个 daemon 线程：主程序退出时不阻塞进程退出
        for name in list(self.pipeline_dict.keys()):
            t = Thread(target=self._camera_thread, args=(name,), daemon=True, name=f"cam-{name}")
            t.start()
            self._capture_threads[name] = t

    def _camera_thread(self, name: str):
        """独立线程：持续从单个相机 pipeline 拉帧。

        每个相机独立 ``wait_for_frames``，互不阻塞。timeout 设为
        2× 帧间隔（例如 60fps → 33ms 帧间隔 → 66ms，取整后按
        2× 帧间隔 = int(2000/fps)ms），下限 50ms——超时只是 continue，
        不会卡死线程（例如相机掉线时）。

        产出物：
          * ``frame_queues[name]`` —— 帧字典 {'color', 'timestamp',
            'frame_number', 'device_time'}，供取帧；
          * ``time_stamps[name]``（加锁）—— 时间戳历史，供延迟统计。
        """
        info = self.pipeline_dict[name]
        cam_cfg = self.camera_configs.get(name, {})
        fps = cam_cfg.get('fps', 30)
        timeout_ms = max(50, int(2000.0 / fps))  # 2× frame interval, min 50ms

        while self.running:
            try:
                frames = info['pipeline'].wait_for_frames(timeout_ms=timeout_ms)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                timestamp = time.time()
                frame_data = {
                    # 零拷贝视图：np.asanyarray 直接引用 SDK 帧的缓冲区，
                    # 不做拷贝（省一次全图 memcpy）。代价是底层缓冲区
                    # 随 SDK 帧生命周期管理——这是官方示例的标准用法，
                    # 实测稳定；若遇到数据乱掉的诡异问题，可改成
                    # np.asarray(color_frame.get_data()).copy()。
                    'color': np.asanyarray(color_frame.get_data()),
                    'timestamp': timestamp,                          # 本机墙钟（秒）
                    'frame_number': color_frame.get_frame_number(),  # SDK 帧序号
                    'device_time': color_frame.timestamp             # 相机硬件时间戳
                }

                self.frame_queues[name].append(frame_data)

                # 时间戳历史加锁：get_timestamp_history() 会遍历这个
                # deque，遍历与 append 并发会抛 RuntimeError，故锁起来。
                with self.lock:
                    self.time_stamps[name].append({
                        'frame_number': color_frame.get_frame_number(),
                        'timestamp': timestamp,
                        'device_time': color_frame.timestamp
                    })

            except Exception as e:
                # 超时、掉线、驱动瞬时错误都吞掉继续跑，
                # 用 debug 级别避免正常超时刷屏
                self.logger.debug(f"相机 {name} 获取帧失败: {e}")
                continue

    def get_latest_frame(self, camera_name: str) -> Optional[Dict[str, Any]]:
        """获取该相机最新一帧。

        **注意实际语义**：``deque.pop()`` 从右侧弹出——右侧正是最新
        append 的帧，因此返回值是"最新帧"，且**会把它移出队列**
        （O(1)）。源码里旧注释写"获取最新但不弹出"是笔误，如果确实
        需要"只看不取"，应改用 ``queue[-1]`` 下标访问。

        未检测到该相机 / 队列为空时返回 None（env.py 会跳过并计数）。
        """
        queue = self.frame_queues.get(camera_name)
        if queue:
            try:
                return queue.pop()# [-1] # 获取最新但不弹出
            except IndexError:
                return None
        return None

    def get_all_latest_frames(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """获取所有相机的最新帧（一次调用拿齐全部相机）。

        env.py 每步 step 调一次：返回 {"cam_high": {...}, ...}，
        某台相机没数据时对应值是 None（上层自行处理缺失）。
        """
        return {name: self.get_latest_frame(name) for name in self.frame_queues}

    def get_timestamp_history(self, camera_name: str) -> List[Dict[str, Any]]:
        """获取该相机的时间戳历史（最近 100 条，按时间顺序）。

        给 env.py 做相机延迟诊断：对比 'timestamp'（本机墙钟）与
        'device_time'（硬件时间戳）可算出采集链路的端到端延迟。
        """
        history = self.time_stamps.get(camera_name)
        if history:
            with self.lock:
                return list(history)
        return []

    def stop_capture(self):
        """停止采集并释放资源（幂等，未启动时调用无副作用）。

        顺序：
          1. running=False → 各采集线程 while 条件失效退出；
          2. join 每个线程（最多等 1 秒——线程正卡在 wait_for_frames
             里时最多等一个 timeout 周期）；
          3. 逐个 pipeline.stop() 释放设备；
          4. 清空 pipeline_dict。

        之后可再次 start_capture() 重新开始（会重新 setup_pipelines）。
        """
        self.running = False
        for name, t in self._capture_threads.items():
            t.join(timeout=1.0)
        self._capture_threads.clear()

        for name, info in self.pipeline_dict.items():
            try:
                info['pipeline'].stop()
                self.logger.info(f"相机 {name} 已停止")
            except Exception as e:
                self.logger.error(f"停止相机 {name} 失败: {e}")

        self.pipeline_dict.clear()

    # --- 上下文管理器协议：with MultiCameraManager(...) as cm: ... ---
    def __enter__(self):
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()

    def __del__(self):
        """析构兜底：忘记显式 stop 时（如异常路径、REPL 交互）也释放设备。

        属"尽力而为"的保险——正常代码请显式调用 stop_capture() 或
        使用 with 语法。
        """
        self.stop_capture()


# ============================================================================
# 冒烟测试入口
# ============================================================================

if __name__ == "__main__":
    # 直接运行 `python -m tron2_env.camera` 即可冒烟测试：
    # 连上相机 → 每 10ms 取一轮最新帧 → 各相机存一张 PNG 到 data/ 目录。
    # 注意：下面的序列号是占位符，需要替换成你手上相机的真实序列号。
    logging.basicConfig(level=logging.INFO)
    serial_to_name = {
        "HEAD_CAMERA_SERIAL": "cam_high",
        "LEFT_WRIST_CAMERA_SERIAL": "cam_left_wrist",
        "RIGHT_WRIST_CAMERA_SERIAL": "cam_right_wrist",
    }
    with MultiCameraManager(serial_to_name=serial_to_name) as cm:
        time.sleep(2)   # 给 pipeline 两秒时间启动、出首帧
        for _ in range(10):
            frames = cm.get_all_latest_frames()
            for name, data in frames.items():
                if data:
                    from PIL import Image
                    # 帧是 BGR 格式，[::-1] 翻转通道变 RGB 再交给 PIL
                    img = Image.fromarray(data['color'][:, :, ::-1])
                    img.save(f"data/{name}.png")
                    print(f"{name}: #{data['frame_number']} @ {data['timestamp']:.3f}")
            time.sleep(0.01)
