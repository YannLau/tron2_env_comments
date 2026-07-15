"""Replay recorded TRON2 actions from a parquet file.

This example is intentionally independent from the policy server. It reads a
recorded trajectory and sends the recorded joint/gripper targets directly to the
robot through ``tron2_env``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any

import numpy as np

from tron2_env import Tron2Config
from tron2_env import create_motion_controller


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_CONFIG = "tron2_openpi/configs/deploy/tron2_deploy.local.yaml"
DEFAULT_INIT_JOINTS = [
    0.026899,
    0.2612,
    -0.02709991,
    -1.5477003,
    0.265,
    0.0180999,
    -0.0614999,
    0.008999,
    -0.269,
    0.02069998,
    -1.5567001,
    -0.254,
    -0.02309972,
    0.06469989,
]
DEFAULT_INIT_HEAD = [1.0467, -0.0139998]


class ParquetReplay:
    def __init__(self, file_path: str | Path, data_key: str):
        self.file_path = Path(file_path).expanduser()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Replay file not found: {self.file_path}")

        self.data_key = data_key
        pl = _import_polars()
        self.df = pl.read_parquet(self.file_path, columns=[data_key])
        if data_key not in self.df.columns:
            raise ValueError(f"Column {data_key!r} not found in {self.file_path}")

    def __len__(self) -> int:
        return self.df.height

    def action(self, index: int) -> np.ndarray:
        value = self.df[self.data_key][index]
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        action = np.asarray(value, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] not in {16, 18}:
            raise ValueError(
                f"{self.data_key}[{index}] must be a 16/18-dim vector, got shape {action.shape}"
            )
        return action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a TRON2 parquet trajectory.")
    parser.add_argument("--file", required=True, help="Path to a .parquet replay file.")
    parser.add_argument(
        "--deploy-config",
        default=DEFAULT_DEPLOY_CONFIG,
        help="Deployment YAML used for robot IP, init pose, backend, and rates.",
    )
    parser.add_argument("--ip", default=None, help="Override robot.ip from the deploy YAML.")
    parser.add_argument(
        "--data-key",
        default="observation.state",
        choices=["observation.state", "action"],
        help="Parquet column to replay. The legacy behavior is observation.state.",
    )
    parser.add_argument("--start-step", type=int, default=0, help="First row to replay.")
    parser.add_argument("--end-step", type=int, default=None, help="Stop before this row.")
    parser.add_argument("--fps", type=float, default=None, help="Override client.fps from the deploy YAML.")
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.5,
        help="Skip a row if any arm joint jumps more than this value from the previous target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the file without moving the robot.",
    )
    return parser.parse_args()


def _import_polars():
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Replay requires polars. Install it with: python -m pip install -e ".[replay]"'
        ) from exc
    return pl


def _import_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Replay requires PyYAML. Install it with: python -m pip install -e ".[replay]"'
        ) from exc
    return yaml


def _resolve_config_path(path: str | Path) -> Path:
    profile_path = Path(path).expanduser()
    if profile_path.is_absolute():
        return profile_path

    candidates = (
        Path.cwd() / profile_path,
        REPO_ROOT / profile_path,
        REPO_ROOT.parent / profile_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_deploy_config(path: str | Path) -> dict[str, Any]:
    resolved_path = _resolve_config_path(path)
    yaml = _import_yaml()
    with resolved_path.open() as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Deploy config must be a mapping: {resolved_path}")
    return data


def _section(profile: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = profile.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Deploy config section must be a mapping: {name}")
    return value


def _robot_config(profile: Mapping[str, Any], robot_ip: str | None) -> Tron2Config:
    robot_profile = _section(profile, "robot")
    init_joints = robot_profile.get("init_joints") or DEFAULT_INIT_JOINTS
    init_head = robot_profile.get("init_head") or DEFAULT_INIT_HEAD

    return Tron2Config(
        robot_ip=str(robot_ip or robot_profile.get("ip", "ROBOT_IP")),
        port=int(robot_profile.get("port", 5000)),
        init_joints=init_joints,
        init_head=init_head,
        state_queue_maxlen=int(robot_profile.get("state_queue_maxlen", 7)),
        polling_rate=float(robot_profile.get("polling_rate", 200.0)),
        connection_timeout=float(robot_profile.get("connection_timeout", 5.0)),
    )


def _full_servo_action(action: np.ndarray, fallback_head: np.ndarray) -> np.ndarray:
    left_arm = action[:7]
    right_arm = action[8:15]
    head = action[16:18] if len(action) >= 18 else fallback_head
    return np.concatenate([left_arm, right_arm, head]).astype(np.float64)


def _arm_action(action: np.ndarray) -> np.ndarray:
    return np.concatenate([action[:7], action[8:15]]).astype(np.float64)


def main() -> None:
    args = _parse_args()
    profile = _load_deploy_config(args.deploy_config)
    client_profile = _section(profile, "client")
    robot_profile = _section(profile, "robot")

    replay = ParquetReplay(args.file, args.data_key)
    start_step = max(0, args.start_step)
    end_step = len(replay) if args.end_step is None else min(args.end_step, len(replay))
    if start_step >= end_step:
        raise ValueError(f"Invalid replay range: start_step={start_step}, end_step={end_step}")

    fps = float(args.fps if args.fps is not None else client_profile.get("fps", 30.0))
    sleep_s = 1.0 / max(fps, 1e-6)
    robot_config = _robot_config(profile, args.ip)
    control_backend = str(
        client_profile.get("control_backend", robot_profile.get("control_backend", "websocket"))
    )
    publish_rate = float(client_profile.get("publish_rate", robot_profile.get("publish_rate", 300.0)))

    print(
        f"Replay file: {replay.file_path} rows={len(replay)} "
        f"range=[{start_step}, {end_step}) key={args.data_key}"
    )
    print(
        f"Robot: {robot_config.robot_ip}:{robot_config.port} "
        f"backend={control_backend} fps={fps:.1f} publish_rate={publish_rate:.1f}"
    )

    if args.dry_run:
        for step in range(start_step, min(end_step, start_step + 3)):
            action = replay.action(step)
            print(
                f"step={step} action_shape={action.shape} "
                f"arm_head={_full_servo_action(action, np.array(DEFAULT_INIT_HEAD)).shape}"
            )
        print("Dry run OK; no robot commands were sent.")
        return

    fallback_head = np.asarray(robot_config.init_head or DEFAULT_INIT_HEAD, dtype=np.float64)
    last_arm_action = np.asarray(robot_config.init_joints or DEFAULT_INIT_JOINTS, dtype=np.float64)

    with create_motion_controller(
        robot_config,
        backend=control_backend,
        publish_rate=publish_rate,
        eta_default=sleep_s,
    ) as robot:
        for step in range(start_step, end_step):
            action = replay.action(step)
            current_arm_action = _arm_action(action)
            error = np.abs(current_arm_action - last_arm_action)
            max_diff = float(np.max(error))
            joint_id = int(np.argmax(error))

            if max_diff >= args.max_joint_delta:
                print(f"step={step}: skip joint {joint_id}, delta={max_diff:.4f}")
                continue

            gripper = np.clip(
                np.array([action[7], action[15]], dtype=np.float64) * 100.0,
                0.0,
                100.0,
            )
            robot.set_gripper(left_opening=float(gripper[0]), right_opening=float(gripper[1]))
            robot.command_joints(_full_servo_action(action, fallback_head))
            last_arm_action = current_arm_action
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
