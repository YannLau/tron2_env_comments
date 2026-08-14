"""Send a single movej joint command to the TRON2 robot and exit.

本示例程序的功能是：连接真实 TRON2 机器人，通过 WebSocket 发送**一条**
关节空间插值运动指令（movej），然后断开连接退出。

与 examples/replay_data.py 的差异（两条控制链路的区别）：
- replay_data.py 走"逐帧伺服"链路：create_motion_controller + command_joints，
  轨迹由**本仓库的插值器**在客户端平滑，发布线程按 300Hz 逐帧推送；
- 本脚本走"整段轨迹"链路：直接使用 WebsocketTransport.movej（见
  src/tron2_env/transport/websocket.py），轨迹由**机器人侧控制器**插值，
  本端只发一条指令，发完即退出。

注意：这里刻意**不使用** MotionController。因为 create_motion_controller 会
启动 300Hz 的 servoj 发布循环，持续把"当前关节角"作为伺服目标发给机器人，
会与机器人侧正在执行的 movej 轨迹打架。仓库里所有 movej 调用（如
WebsocketTransport._move_to_init_pose）都发生在发布线程启动之前，
本脚本同理，直接用 transport。

使用示例：
    python examples/send_movej.py --ip 192.168.1.100          # 走到默认关节角
    python examples/send_movej.py --joints 0 0.26 -0.03 -1.55 0.27 0.02 -0.06 0.01 -0.27 0.02 -1.56 -0.25 -0.02 0.06 --move-time 3
    python examples/send_movej.py --wait                      # 发送后阻塞等待到位再退出
"""

from __future__ import annotations  # 启用延迟求值注解，允许使用尚未定义的类型

import argparse  # 命令行参数解析
from collections.abc import Mapping  # 映射类型的抽象基类，用于类型检查
from pathlib import Path  # 面向对象的文件系统路径处理
from typing import Any  # 任意类型，用于灵活的类型注解

import numpy as np  # 数值计算库，处理关节向量

# 从 tron2_env 库导入核心组件
from tron2_env import Tron2Config  # 机器人连接配置的数据类
from tron2_env import WebsocketTransport  # WebSocket 传输层（movej 方法所在）
from tron2_env.joints import JointIndex  # 关节维度/索引常量（用于维度校验）


# ── 路径与默认常量 ────────────────────────────────────────────────────────────
# 计算仓库根目录：当前文件向上两级目录
REPO_ROOT = Path(__file__).resolve().parents[1]

# 默认的部署配置文件路径（与 replay_data.py 保持一致）
DEFAULT_DEPLOY_CONFIG = "tron2_openpi/configs/deploy/tron2_deploy.local.yaml"

# 默认目标关节角：14 维（左臂 7 关节 + 右臂 7 关节）
# 与 replay_data.py 的 DEFAULT_INIT_JOINTS 相同，是安全的家位姿态
DEFAULT_JOINTS = [
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


# ── 命令行参数解析 ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Send one movej joint command to the TRON2 robot and exit."
    )

    # --deploy-config：部署配置文件路径（用于读取机器人 IP、端口等）
    parser.add_argument(
        "--deploy-config",
        default=DEFAULT_DEPLOY_CONFIG,
        help="Deployment YAML used for robot IP and port.",
    )

    # --ip：覆盖部署配置中的机器人 IP 地址
    parser.add_argument("--ip", default=None, help="Override robot.ip from the deploy YAML.")

    # --joints：目标关节角（14 维：左臂 7 关节 + 右臂 7 关节，单位弧度）
    # 注意：movej 只接受手臂关节（MOVEJ_DIM=14），不含夹爪和头部
    # nargs="+" 允许多个值；argparse 能识别 -0.03 这类以负号开头的数值
    parser.add_argument(
        "--joints",
        type=float,
        nargs="+",
        default=None,
        help="Target arm joints (14 values: 7 left + 7 right, radians). "
        "Defaults to the safe home pose.",
    )

    # --move-time：机器人侧插值走完轨迹的时间（秒）
    parser.add_argument(
        "--move-time",
        type=float,
        default=2.0,
        help="Duration of the robot-side interpolation (seconds).",
    )

    # --wait：发送后不立即退出，而是阻塞等待机器人到位（默认立即退出）
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the robot reaches the target before exiting.",
    )

    # --timeout：--wait 模式下的最长等待时间（秒）
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Max wait time for --wait (seconds).",
    )

    return parser.parse_args()


# ── 配置文件加载 ─────────────────────────────────────────────────────────────
# 以下三个函数与 replay_data.py 同名函数一致，用于从 YAML 中读取连接参数。

def _import_yaml():
    """延迟导入 PyYAML 库（用于解析部署配置文件）。"""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'This script requires PyYAML. Install it with: python -m pip install -e ".[replay]"'
        ) from exc
    return yaml


def _resolve_config_path(path: str | Path) -> Path:
    """将配置文件路径解析为实际存在的绝对路径。

    查找顺序：当前工作目录 → 仓库根目录 → 仓库上级目录。
    """
    profile_path = Path(path).expanduser()
    if profile_path.is_absolute():
        return profile_path

    candidates = (
        Path.cwd() / profile_path,
        REPO_ROOT / profile_path,
        REPO_ROOT.parent / profile_path,
        REPO_ROOT.parent.parent / profile_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_deploy_config(path: str | Path) -> dict[str, Any]:
    """加载并解析部署配置 YAML 文件。"""
    resolved_path = _resolve_config_path(path)
    yaml = _import_yaml()
    with resolved_path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Deploy config must be a mapping: {resolved_path}")
    return data


def _section(profile: Mapping[str, Any], name: str) -> dict[str, Any]:
    """从配置字典中安全地提取一个子配置段（如 "robot"）。"""
    value = profile.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Deploy config section must be a mapping: {name}")
    return value


# ── 机器人配置 ────────────────────────────────────────────────────────────────

def _transport_config(profile: Mapping[str, Any], robot_ip: str | None) -> Tron2Config:
    """从部署配置中构建 Tron2Config 对象。

    与 replay_data.py 的 _robot_config 的关键区别：
    这里**刻意不传** init_joints / init_head。因为 WebsocketTransport 的
    构造函数在检测到初始位姿时会先执行 _move_to_init_pose()（等于多发一次
    移动），与本脚本"只发一条 movej 指令"的目标冲突，因此留空跳过。
    """
    robot_profile = _section(profile, "robot")

    return Tron2Config(
        robot_ip=str(robot_ip or robot_profile.get("ip", "ROBOT_IP")),
        port=int(robot_profile.get("port", 5000)),
        # 不传 init_joints / init_head：跳过构造函数里的 _move_to_init_pose
        state_queue_maxlen=int(robot_profile.get("state_queue_maxlen", 7)),
        polling_rate=float(robot_profile.get("polling_rate", 200.0)),
        connection_timeout=float(robot_profile.get("connection_timeout", 5.0)),
    )


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """程序入口：解析参数、加载配置、发送一条 movej 指令后退出。

    执行流程：
    1. 解析命令行参数
    2. 加载部署配置文件（YAML），构建 Tron2Config
    3. 连接机器人（WebsocketTransport 构造函数会建立连接并启动状态轮询）
    4. 读取并打印当前关节状态（仅作信息展示，失败不阻塞）
    5. 发送一条 movej 指令；若指定 --wait 则等待到位后再退出
    """
    # ── 步骤 1：解析命令行参数 ──
    args = _parse_args()

    # ── 步骤 2：加载配置 ──
    profile = _load_deploy_config(args.deploy_config)
    transport_config = _transport_config(profile, args.ip)

    # ── 步骤 3：校验目标关节角 ──
    # movej 只接受 14 维手臂关节（见 websocket.py 中 MOVEJ_DIM 校验）
    target = np.asarray(args.joints or DEFAULT_JOINTS, dtype=np.float64)
    if target.shape[0] != JointIndex.MOVEJ_DIM:
        raise ValueError(
            f"--joints must have {JointIndex.MOVEJ_DIM} values "
            f"(7 left + 7 right), got {target.shape[0]}"
        )

    print(
        f"Robot: {transport_config.robot_ip}:{transport_config.port} "
        f"move_time={args.move_time:.2f}s wait={args.wait}"
    )

    # ── 步骤 4：连接 → 发送指令 → 退出 ──
    # WebsocketTransport 支持 with 语法：块结束时自动 disconnect()。
    # 构造时只建立连接 + 启动状态轮询，不执行任何移动（见 _transport_config）。
    with WebsocketTransport(transport_config) as transport:
        # 打印当前关节状态（仅信息展示；读不到也不影响发指令）
        try:
            state = transport.get_joint_state(timeout=2.0)
            current_arm = np.concatenate(
                [state["states"][JointIndex.LEFT_ARM], state["states"][JointIndex.RIGHT_ARM]]
            )
            print(f"Current arm joints: {np.array2string(current_arm, precision=4)}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read current state: {exc}")

        # 发送唯一一条指令：机器人侧用 move_time 秒插值到目标关节角
        print(
            f"movej target: {np.array2string(target, precision=4)} "
            f"(move_time={args.move_time:.2f}s)"
        )
        transport.movej(target, move_time=args.move_time)
        print("movej sent.")

        # --wait 模式：阻塞等待到位（默认发完即退出）
        if args.wait:
            print(f"Waiting up to {args.timeout:.1f}s for the robot to reach the target...")
            reached = transport.wait_until_reached(
                target, tolerance=0.05, timeout=args.timeout
            )
            print("Robot reached target." if reached else "Timeout: robot did not reach target.")

    print("Done; connection closed.")


# ── 程序入口 ──
if __name__ == "__main__":
    main()
