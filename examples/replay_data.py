"""Replay recorded TRON2 actions from a parquet file.

本示例程序的功能是：从 Parquet 文件中读取预先录制的机器人轨迹数据，然后将这些
动作指令重新发送给真实的 TRON2 机器人进行回放。

这个示例是**独立于策略服务器（policy server）** 设计的。它直接从录制文件中读取
关节目标值（joint targets）和夹爪目标值（gripper targets），然后通过
``tron2_env`` 库将指令发送给机器人执行。

核心流程：
1. 读取 Parquet 文件中的指定列（默认为 observation.state）
2. 从部署配置 YAML 文件中加载机器人的连接参数（IP、端口等）
3. 逐帧解析动作向量，过滤掉关节变化过大的异常帧
4. 通过运动控制器将关节角度和夹爪开度发送给机器人

使用示例：
    python examples/replay_data.py --file trajectory.parquet --dry-run  # 仅验证不执行
    python examples/replay_data.py --file trajectory.parquet --fps 30  # 以30FPS回放
"""

from __future__ import annotations  # 启用延迟求值注解，允许使用尚未定义的类型

import argparse  # 命令行参数解析
from collections.abc import Mapping  # 映射类型的抽象基类，用于类型检查
from pathlib import Path  # 面向对象的文件系统路径处理
import time  # 时间相关操作（sleep 用于控制回放帧率）
from typing import Any  # 任意类型，用于灵活的类型注解

import numpy as np  # 数值计算库，处理动作向量

# 从 tron2_env 库导入核心组件
from tron2_env import Tron2Config  # 机器人连接配置的数据类
from tron2_env import create_motion_controller  # 创建运动控制器的工厂函数


# ── 路径与默认常量 ────────────────────────────────────────────────────────────
# 计算仓库根目录：当前文件向上两级目录
# __file__ → examples/replay_data.py → parents[1] → 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]

# 默认的部署配置文件路径（相对于仓库根目录）
# YAML 文件包含机器人 IP、初始化姿态、后端类型、控制频率等信息
DEFAULT_DEPLOY_CONFIG = "tron2_openpi/configs/deploy/tron2_deploy.local.yaml"

# 默认的 14 个关节初始角度（单位：弧度）
# 这些值是 TRON2 机器人双臂各 7 个关节的安全起始位置
# 索引对应关系：
#   前 7 个值 → 左臂关节 (left arm joints)
#   后 7 个值 → 右臂关节 (right arm joints)
# 注意：这里的关节顺序与动作向量中的顺序不同（动作向量是左右臂交替排列的）
DEFAULT_INIT_JOINTS = [
    0.026899,    # 左臂关节1
    0.2612,      # 左臂关节2
    -0.02709991, # 左臂关节3
    -1.5477003,  # 左臂关节4
    0.265,       # 左臂关节5
    0.0180999,   # 左臂关节6
    -0.0614999,  # 左臂关节7
    0.008999,    # 右臂关节1
    -0.269,      # 右臂关节2
    0.02069998,  # 右臂关节3
    -1.5567001,  # 右臂关节4
    -0.254,      # 右臂关节5
    -0.02309972, # 右臂关节6
    0.06469989,  # 右臂关节7
]

# 默认的头部关节初始角度（单位：弧度）
# 两个值分别对应头部的两个自由度（俯仰和偏航）
DEFAULT_INIT_HEAD = [1.0467, -0.0139998]


# ── ParquetReplay 类 ───────────────────────────────────────────────────────────

class ParquetReplay:
    """从 Parquet 文件中读取并解析 TRON2 机器人动作轨迹。

    这个类封装了 Parquet 文件的读取逻辑，提供按索引访问动作向量的接口。
    每一行数据对应轨迹中的一个时间步。

    Parameters
    ----------
    file_path : str or Path
        Parquet 文件的路径，支持 ~ 展开为用户主目录。
    data_key : str
        要读取的列名，通常为 "observation.state" 或 "action"。
        - "observation.state"：录制的观测状态（关节角度等）
        - "action"：录制的动作指令

    Raises
    ------
    FileNotFoundError
        如果指定的 Parquet 文件不存在。
    ValueError
        如果指定的列名在文件中不存在。
    """

    def __init__(self, file_path: str | Path, data_key: str):
        # 将路径转为 Path 对象并展开 ~（如果用户传了 ~/data/file.parquet）
        self.file_path = Path(file_path).expanduser()

        # 检查文件是否存在，不存在则立即报错（尽早失败原则）
        if not self.file_path.exists():
            raise FileNotFoundError(f"Replay file not found: {self.file_path}")

        self.data_key = data_key  # 记录要读取的列名

        # 延迟导入 polars：只在真正需要时才导入
        # 这样做的好处是：如果用户只是想看帮助信息（--help），不需要安装 polars
        pl = _import_polars()

        # 使用 polars 读取 Parquet 文件，只读取需要的列以节省内存
        # columns=[data_key] 表示只加载指定的列，忽略其他所有列
        self.df = pl.read_parquet(self.file_path, columns=[data_key])

        # 校验列名是否存在
        if data_key not in self.df.columns:
            raise ValueError(f"Column {data_key!r} not found in {self.file_path}")

    def __len__(self) -> int:
        """返回轨迹的总帧数（行数）。"""
        # polars DataFrame 的 height 属性返回行数
        return self.df.height

    def action(self, index: int) -> np.ndarray:
        """获取指定索引处的动作向量。

        从 Parquet 文件的指定行中提取动作数据，将其转换为 NumPy 数组。

        Parameters
        ----------
        index : int
            要获取的行索引（0-based）。

        Returns
        -------
        np.ndarray
            形状为 (16,) 或 (18,) 的一维 float32 数组。

            16 维动作向量的结构：
            - [0:7]   → 左臂 7 个关节角度
            - [7]     → 左夹爪开度（0-1 范围）
            - [8:15]  → 右臂 7 个关节角度
            - [15]    → 右夹爪开度（0-1 范围）

            18 维动作向量在尾部多了 2 个元素：
            - [16:18] → 头部关节角度（俯仰、偏航）

        Raises
        ------
        ValueError
            如果动作向量的维度不是 16 或 18。
        """
        # 从 DataFrame 的指定行和列中取出原始值
        value = self.df[self.data_key][index]

        # 如果值是 polars 的 Series/Array 类型，先转为 NumPy 数组
        # 这是因为 Parquet 中的 list/array 列在 polars 中可能是嵌套类型
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()

        # 确保动作向量是 float32 类型的一维数组
        action = np.asarray(value, dtype=np.float32)

        # 校验动作向量的维度
        # 16 维：双臂关节 + 夹爪（无头部）
        # 18 维：双臂关节 + 夹爪 + 头部关节
        if action.ndim != 1 or action.shape[0] not in {16, 18}:
            raise ValueError(
                f"{self.data_key}[{index}] must be a 16/18-dim vector, got shape {action.shape}"
            )
        return action


# ── 命令行参数解析 ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回的命名空间包含所有回放所需的配置参数，用户可以通过命令行灵活控制
    回放行为（文件路径、帧率、IP 地址、过滤阈值等）。

    Returns
    -------
    argparse.Namespace
        解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Replay a TRON2 parquet trajectory."
    )

    # --file：必选参数，指定要回放的 Parquet 轨迹文件路径
    parser.add_argument("--file", required=True, help="Path to a .parquet replay file.")

    # --deploy-config：部署配置文件路径（YAML 格式）
    # 该文件包含机器人的 IP、端口、初始姿态等配置
    parser.add_argument(
        "--deploy-config",
        default=DEFAULT_DEPLOY_CONFIG,
        help="Deployment YAML used for robot IP, init pose, backend, and rates.",
    )

    # --ip：可覆盖部署配置中的机器人 IP 地址
    # 当需要在不同网络环境下使用同一份配置时非常有用
    parser.add_argument("--ip", default=None, help="Override robot.ip from the deploy YAML.")

    # --data-key：指定 Parquet 文件中要回放的列名
    # observation.state：回放录制的观测状态（旧版本的默认行为）
    # action：回放录制的动作指令
    parser.add_argument(
        "--data-key",
        default="observation.state",
        choices=["observation.state", "action"],
        help="Parquet column to replay. The legacy behavior is observation.state.",
    )

    # --start-step：回放的起始帧索引（0-based）
    # 可以从轨迹的中间某帧开始回放，方便调试或跳过不需要的部分
    parser.add_argument("--start-step", type=int, default=0, help="First row to replay.")

    # --end-step：回放的结束帧索引（不包含该帧）
    # 为 None 表示回放到轨迹末尾
    parser.add_argument("--end-step", type=int, default=None, help="Stop before this row.")

    # --fps：回放的帧率（Frames Per Second）
    # 覆盖部署配置中的 client.fps 设置
    # 控制每两帧之间的等待时间 = 1/fps 秒
    parser.add_argument("--fps", type=float, default=None, help="Override client.fps from the deploy YAML.")

    # --max-joint-delta：关节变化阈值（单位：弧度）
    # 如果某一帧中任一关节角度的变化超过此阈值，则跳过该帧
    # 这是安全机制：防止回放时因数据异常导致机器人剧烈运动
    # 默认 0.5 弧度 ≈ 28.6 度
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.5,
        help="Skip a row if any arm joint jumps more than this value from the previous target.",
    )

    # --dry-run：干运行模式
    # 开启后只加载和验证数据文件，不会向机器人发送任何运动指令
    # 建议在每次回放前先执行一次 dry-run 确认数据正确
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the file without moving the robot.",
    )

    return parser.parse_args()


# ── 延迟导入辅助函数 ──────────────────────────────────────────────────────────
# 这些函数实现了"懒加载"模式：只有在真正需要使用时才导入第三方库。
# 好处：
# 1. 加快了程序启动速度（import 阶段不需要加载这些库）
# 2. 用户运行 --help 时不需要安装这些依赖
# 3. 出错时能给出友好的安装提示

def _import_polars():
    """延迟导入 polars 库（用于读取 Parquet 文件）。

    Returns
    -------
    module
        polars 模块对象。

    Raises
    ------
    RuntimeError
        如果 polars 未安装，给出安装命令提示。
    """
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Replay requires polars. Install it with: python -m pip install -e ".[replay]"'
        ) from exc
    return pl


def _import_yaml():
    """延迟导入 PyYAML 库（用于解析部署配置文件）。

    Returns
    -------
    module
        yaml 模块对象。

    Raises
    ------
    RuntimeError
        如果 PyYAML 未安装，给出安装命令提示。
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Replay requires PyYAML. Install it with: python -m pip install -e ".[replay]"'
        ) from exc
    return yaml


# ── 配置文件加载 ─────────────────────────────────────────────────────────────

def _resolve_config_path(path: str | Path) -> Path:
    """将配置文件路径解析为实际存在的绝对路径。

    查找顺序（按优先级从高到低）：
    1. 如果传入的是绝对路径，直接使用
    2. 当前工作目录下的相对路径
    3. 仓库根目录（REPO_ROOT）下的相对路径
    4. 仓库上级目录下的相对路径

    如果所有候选路径都不存在，返回当前工作目录下的路径（让后续代码报错）。

    Parameters
    ----------
    path : str or Path
        用户指定的配置文件路径（可以是相对或绝对路径）。

    Returns
    -------
    Path
        解析后的路径（优先返回存在的）。
    """
    profile_path = Path(path).expanduser()

    # 如果是绝对路径，直接返回（无需查找）
    if profile_path.is_absolute():
        return profile_path

    # 按优先级排列的候选路径列表
    candidates = (
        Path.cwd() / profile_path,        # 当前目录
        REPO_ROOT / profile_path,         # 仓库根目录
        REPO_ROOT.parent / profile_path,  # 仓库上级目录
    )

    # 返回第一个存在的候选路径
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 如果都不存在，返回第一个候选路径，让后续打开文件时报错
    return candidates[0]


def _load_deploy_config(path: str | Path) -> dict[str, Any]:
    """加载并解析部署配置 YAML 文件。

    Parameters
    ----------
    path : str or Path
        配置文件路径。

    Returns
    -------
    dict
        解析后的配置字典。如果 YAML 文件为空，返回空字典。

    Raises
    ------
    ValueError
        如果配置文件顶层不是字典（映射）类型。
    """
    resolved_path = _resolve_config_path(path)
    yaml = _import_yaml()

    # 使用 safe_load 而不是 load，避免 YAML 中的任意代码执行
    with resolved_path.open() as f:
        data = yaml.safe_load(f) or {}  # safe_load 对空文件返回 None，用 or {} 兜底

    # 校验顶层结构必须是字典
    if not isinstance(data, dict):
        raise ValueError(f"Deploy config must be a mapping: {resolved_path}")
    return data


def _section(profile: Mapping[str, Any], name: str) -> dict[str, Any]:
    """从配置字典中安全地提取一个子配置段。

    部署 YAML 的典型结构：
        robot:
          ip: "192.168.1.100"
          port: 5000
          init_joints: [...]
        client:
          fps: 30.0
          control_backend: "websocket"

    这个函数用于提取 robot 或 client 这样的子段，
    如果子段不存在或为 None，返回空字典。

    Parameters
    ----------
    profile : Mapping
        完整的配置字典。
    name : str
        要提取的子段名称（如 "robot"、"client"）。

    Returns
    -------
    dict
        子段配置字典。如果不存在则返回 {}。

    Raises
    ------
    ValueError
        如果子段存在但不是字典类型。
    """
    value = profile.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Deploy config section must be a mapping: {name}")
    return value


# ── 机器人配置 ────────────────────────────────────────────────────────────────

def _robot_config(profile: Mapping[str, Any], robot_ip: str | None) -> Tron2Config:
    """从部署配置中构建 Tron2Config 对象。

    合并 YAML 配置、命令行覆盖参数和代码默认值，生成完整的机器人连接配置。

    Parameters
    ----------
    profile : Mapping
        完整的部署配置字典。
    robot_ip : str or None
        命令行指定的 IP 地址覆盖。为 None 时使用 YAML 中的值。

    Returns
    -------
    Tron2Config
        机器人连接配置对象，包含 IP、端口、初始姿态等全部参数。
    """
    robot_profile = _section(profile, "robot")

    # 初始关节角度：优先使用 YAML 配置，否则用代码默认值
    init_joints = robot_profile.get("init_joints") or DEFAULT_INIT_JOINTS
    # 初始头部角度：优先使用 YAML 配置，否则用代码默认值
    init_head = robot_profile.get("init_head") or DEFAULT_INIT_HEAD

    return Tron2Config(
        # IP 地址：命令行参数 > YAML 配置 > 占位符 "ROBOT_IP"
        robot_ip=str(robot_ip or robot_profile.get("ip", "ROBOT_IP")),
        # 端口号：默认 5000
        port=int(robot_profile.get("port", 5000)),
        init_joints=init_joints,
        init_head=init_head,
        # 状态队列最大长度：用于平滑状态估计，默认 7
        state_queue_maxlen=int(robot_profile.get("state_queue_maxlen", 7)),
        # 状态轮询频率（Hz）：默认 200 Hz
        polling_rate=float(robot_profile.get("polling_rate", 200.0)),
        # 连接超时时间（秒）：默认 5 秒
        connection_timeout=float(robot_profile.get("connection_timeout", 5.0)),
    )


# ── 动作向量转换函数 ─────────────────────────────────────────────────────────
# 这些函数负责在"录制格式"和"机器人控制格式"之间进行转换。
#
# 录制的动作向量格式（Parquet 中的原始数据）：
#   [左臂7关节] [左夹爪] [右臂7关节] [右夹爪] [头部2关节(可选)]
#   共 16 维（无头部）或 18 维（含头部）
#
# 机器人控制需要的格式：
#   - 关节伺服控制（full_servo）：[左臂7关节] [右臂7关节] [头部2关节] = 16 维
#   - 关节安全检查（arm_action）：  [左臂7关节] [右臂7关节] = 14 维

def _full_servo_action(action: np.ndarray, fallback_head: np.ndarray) -> np.ndarray:
    """将录制的动作向量转换为完整的伺服控制指令（含头部）。

    从 16 或 18 维的动作向量中提取双臂关节角度和头部角度，
    组装成机器人伺服控制所需的 16 维向量。

    提取逻辑：
    - action[0:7]   → 左臂关节（7个）
    - action[8:15]  → 右臂关节（7个，跳过索引7的左夹爪）
    - action[16:18] → 头部关节（如果存在）；否则使用 fallback_head

    Parameters
    ----------
    action : np.ndarray
        原始动作向量（16 或 18 维）。
    fallback_head : np.ndarray
        当动作向量不包含头部数据时使用的默认头部角度（2维）。

    Returns
    -------
    np.ndarray
        16 维 float64 数组：[左臂7关节, 右臂7关节, 头部2关节]
    """
    # 左臂关节：索引 0-6（共 7 个）
    left_arm = action[:7]
    # 右臂关节：索引 8-14（共 7 个），跳过索引 7（左夹爪）
    right_arm = action[8:15]
    # 头部关节：如果动作向量 >= 18 维则取尾部 2 个，否则用默认值
    head = action[16:18] if len(action) >= 18 else fallback_head

    # 拼接为完整的伺服控制指令，转换为 float64（机器人控制需要高精度）
    return np.concatenate([left_arm, right_arm, head]).astype(np.float64)


def _arm_action(action: np.ndarray) -> np.ndarray:
    """从动作向量中提取双臂关节角度（不含夹爪和头部）。

    用于关节变化量检查（max_joint_delta 过滤），因为夹爪的变化量
    不适合用角度阈值来判断。

    Parameters
    ----------
    action : np.ndarray
        原始动作向量（16 或 18 维）。

    Returns
    -------
    np.ndarray
        14 维 float64 数组：[左臂7关节, 右臂7关节]
    """
    # 左臂关节 + 右臂关节，跳过索引 7（左夹爪）和索引 15（右夹爪）
    return np.concatenate([action[:7], action[8:15]]).astype(np.float64)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """程序入口：解析参数、加载配置、执行轨迹回放。

    执行流程：
    1. 解析命令行参数
    2. 加载部署配置文件（YAML）
    3. 加载 Parquet 轨迹数据
    4. 确定回放范围和帧率
    5. 如果是 dry-run 模式，仅打印前几帧信息后退出
    6. 否则创建运动控制器，逐帧发送指令给机器人
    """
    # ── 步骤 1：解析命令行参数 ──
    args = _parse_args()

    # ── 步骤 2：加载配置 ──
    profile = _load_deploy_config(args.deploy_config)
    # 提取配置的各子段
    client_profile = _section(profile, "client")   # 客户端配置（帧率、后端等）
    robot_profile = _section(profile, "robot")     # 机器人配置（IP、姿态等）

    # ── 步骤 3：加载轨迹数据 ──
    replay = ParquetReplay(args.file, args.data_key)

    # ── 步骤 4：确定回放范围 ──
    # 起始帧：不能小于 0
    start_step = max(0, args.start_step)
    # 结束帧：未指定则到轨迹末尾；指定了则不超过轨迹长度
    end_step = len(replay) if args.end_step is None else min(args.end_step, len(replay))
    # 校验范围的有效性
    if start_step >= end_step:
        raise ValueError(f"Invalid replay range: start_step={start_step}, end_step={end_step}")

    # ── 步骤 5：计算帧间等待时间 ──
    # FPS 优先级：命令行参数 > YAML 配置 > 默认 30.0
    fps = float(args.fps if args.fps is not None else client_profile.get("fps", 30.0))
    # 每帧之间的睡眠时间 = 1/帧率
    # max(fps, 1e-6) 防止除以零（虽然正常情况下 fps 不会为 0）
    sleep_s = 1.0 / max(fps, 1e-6)

    # ── 步骤 6：构建机器人配置 ──
    # robot_config 包含 IP、端口、初始姿态等
    robot_config = _robot_config(profile, args.ip)

    # 控制后端类型：websocket 或 grpc 等
    # 优先级：client.control_backend > robot.control_backend > "websocket"
    control_backend = str(
        client_profile.get("control_backend", robot_profile.get("control_backend", "websocket"))
    )

    # 指令发布频率（Hz）：每秒向机器人发送多少次指令
    # 注意：这是底层控制的频率，与回放帧率（fps）是不同的概念
    publish_rate = float(client_profile.get("publish_rate", robot_profile.get("publish_rate", 300.0)))

    # ── 打印回放信息 ──
    print(
        f"Replay file: {replay.file_path} rows={len(replay)} "
        f"range=[{start_step}, {end_step}) key={args.data_key}"
    )
    print(
        f"Robot: {robot_config.robot_ip}:{robot_config.port} "
        f"backend={control_backend} fps={fps:.1f} publish_rate={publish_rate:.1f}"
    )

    # ── dry-run 模式：仅验证数据，不控制机器人 ──
    if args.dry_run:
        # 打印前几帧（最多 3 帧）的动作向量信息
        for step in range(start_step, min(end_step, start_step + 3)):
            action = replay.action(step)
            print(
                f"step={step} action_shape={action.shape} "
                f"arm_head={_full_servo_action(action, np.array(DEFAULT_INIT_HEAD)).shape}"
            )
        print("Dry run OK; no robot commands were sent.")
        return  # dry-run 模式到此结束，不执行后续的机器人控制

    # ── 实际回放模式 ──

    # 备用头部角度：当动作向量不包含头部数据时使用
    # 优先级：robot_config.init_head > DEFAULT_INIT_HEAD
    fallback_head = np.asarray(robot_config.init_head or DEFAULT_INIT_HEAD, dtype=np.float64)

    # 上一帧的手臂关节角度，用于检测关节变化是否过大
    # 初始值设为机器人的初始姿态
    last_arm_action = np.asarray(robot_config.init_joints or DEFAULT_INIT_JOINTS, dtype=np.float64)

    # ── 创建运动控制器并开始回放 ──
    # create_motion_controller 是一个上下文管理器（with 语句），
    # 进入时连接到机器人，退出时自动断开连接并清理资源
    with create_motion_controller(
        robot_config,
        backend=control_backend,
        publish_rate=publish_rate,
        eta_default=sleep_s,  # 默认的期望到达时间（用于轨迹平滑）
    ) as robot:
        # 逐帧遍历轨迹
        for step in range(start_step, end_step):
            # 获取当前帧的动作向量
            action = replay.action(step)

            # 提取当前帧的双臂关节角度（14 维，不含夹爪和头部）
            current_arm_action = _arm_action(action)

            # ── 安全检查：关节变化量过滤 ──
            # 计算当前帧与上一帧之间每个关节的绝对变化量
            error = np.abs(current_arm_action - last_arm_action)
            max_diff = float(np.max(error))          # 最大变化量
            joint_id = int(np.argmax(error))          # 变化最大的关节索引

            # 如果任一关节的变化超过阈值，跳过该帧
            # 这是防止机器人因数据异常而发生剧烈运动的安全机制
            if max_diff >= args.max_joint_delta:
                print(f"step={step}: skip joint {joint_id}, delta={max_diff:.4f}")
                continue  # 跳过该帧，不更新 last_arm_action

            # ── 夹爪控制 ──
            # 动作向量中夹爪的值在 [0, 1] 范围内，需要乘以 100 转为 [0, 100]
            # action[7]  = 左夹爪开度
            # action[15] = 右夹爪开度
            # np.clip 确保值在 [0, 100] 的安全范围内（0=完全闭合，100=完全张开）
            gripper = np.clip(
                np.array([action[7], action[15]], dtype=np.float64) * 100.0,
                0.0,
                100.0,
            )

            # 设置左右夹爪的开度
            robot.set_gripper(left_opening=float(gripper[0]), right_opening=float(gripper[1]))

            # ── 关节伺服控制 ──
            # 将动作向量转换为完整的伺服指令并发送给机器人
            robot.command_joints(_full_servo_action(action, fallback_head))

            # 更新"上一帧关节角度"为当前帧的值，用于下一轮的变化量检查
            last_arm_action = current_arm_action

            # ── 帧率控制 ──
            # 休眠指定时长以维持目标帧率
            # 注意：这是一个简化的帧率控制方式，
            # 实际的命令发送 + 休眠时间会比 sleep_s 略长
            time.sleep(sleep_s)


# ── 程序入口 ──
# 当直接运行此文件时（python replay_data.py）执行 main()
# 当作为模块导入时（import replay_data）不执行
if __name__ == "__main__":
    main()
