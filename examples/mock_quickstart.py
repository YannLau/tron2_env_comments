"""Run a software-only MotionController example without connecting to a robot."""

from __future__ import annotations

import threading
import time

import numpy as np

from tron2_env.joints import JointIndex
from tron2_env.motion import MotionController


class RecordingTransport:
    """In-memory RobotTransport used only by this no-hardware example."""

    def __init__(self) -> None:
        self._connected = True
        self._lock = threading.Lock()
        self._sent: list[np.ndarray] = []
        self._state = np.zeros(JointIndex.STATE_DIM, dtype=np.float64)

    def send_joint_cmd(self, q: np.ndarray) -> None:
        with self._lock:
            self._sent.append(np.asarray(q, dtype=np.float64).copy())

    def get_joint_state(self, timeout: float = 1.0) -> dict:
        del timeout
        return {"timestamp": int(time.time() * 1000), "states": self._state.tolist()}

    def get_head_position(self) -> np.ndarray:
        return self._state[JointIndex.HEAD].copy()

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        del left_opening, right_opening

    def wait_until_reached(
        self,
        target_joints,
        tolerance: float = 0.05,
        timeout: float = 10.0,
    ) -> bool:
        del target_joints, tolerance, timeout
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        self.disconnect()

    def sent_frames(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._sent)


def main() -> None:
    transport = RecordingTransport()
    controller = MotionController(transport=transport, publish_rate=100.0, eta_default=0.02)
    target = np.linspace(-0.1, 0.1, JointIndex.SERVOJ_DIM)

    controller.start()
    try:
        controller.command_joints(target)
        time.sleep(0.08)
    finally:
        controller.disconnect()

    frames = transport.sent_frames()
    if not frames:
        raise RuntimeError("mock controller did not publish any frames")

    np.testing.assert_allclose(frames[-1], target, atol=1e-6)
    print(f"Mock transport published {len(frames)} frames; no robot connection was opened.")


if __name__ == "__main__":
    main()
