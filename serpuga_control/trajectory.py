"""Reference trajectory generation and horizon previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ReferencePreview:
    poses: np.ndarray
    speeds: np.ndarray
    yaw_rates: np.ndarray


@dataclass
class ReferenceTrajectory:
    """Time-parameterised planar trajectory with feed-forward twist."""

    times: np.ndarray
    poses: np.ndarray
    speeds: np.ndarray
    yaw_rates: np.ndarray

    @classmethod
    def from_twist_profile(
        cls,
        duration: float,
        integration_dt: float,
        speed_function: Callable[[float], float],
        yaw_rate_function: Callable[[float], float],
        initial_pose: np.ndarray | None = None,
    ) -> "ReferenceTrajectory":
        initial = (
            np.zeros(3, dtype=float)
            if initial_pose is None
            else np.asarray(initial_pose, dtype=float)
        )
        times = np.arange(0.0, duration + integration_dt, integration_dt)
        poses = np.zeros((times.size, 3), dtype=float)
        speeds = np.array([speed_function(t) for t in times], dtype=float)
        yaw_rates = np.array([yaw_rate_function(t) for t in times], dtype=float)
        poses[0] = initial
        for index in range(times.size - 1):
            dt = times[index + 1] - times[index]
            yaw_mid = poses[index, 2] + 0.5 * dt * yaw_rates[index]
            poses[index + 1, 0] = poses[index, 0] + dt * speeds[index] * np.cos(yaw_mid)
            poses[index + 1, 1] = poses[index, 1] + dt * speeds[index] * np.sin(yaw_mid)
            poses[index + 1, 2] = poses[index, 2] + dt * yaw_rates[index]
        return cls(times=times, poses=poses, speeds=speeds, yaw_rates=yaw_rates)

    @classmethod
    def straight(
        cls,
        duration: float,
        integration_dt: float,
        speed: float,
    ) -> "ReferenceTrajectory":
        return cls.from_twist_profile(
            duration=duration,
            integration_dt=integration_dt,
            speed_function=lambda _time: speed,
            yaw_rate_function=lambda _time: 0.0,
        )

    @classmethod
    def gentle_turn(
        cls,
        duration: float,
        integration_dt: float,
        speed: float,
        peak_yaw_rate: float = 0.22,
    ) -> "ReferenceTrajectory":
        def yaw_rate(time: float) -> float:
            if time < 0.2 * duration or time > 0.8 * duration:
                return 0.0
            phase = (time - 0.2 * duration) / (0.6 * duration)
            return peak_yaw_rate * np.sin(2.0 * np.pi * phase)

        return cls.from_twist_profile(
            duration=duration,
            integration_dt=integration_dt,
            speed_function=lambda _time: speed,
            yaw_rate_function=yaw_rate,
        )

    def _interpolate_pose(self, query_times: np.ndarray) -> np.ndarray:
        poses = np.empty((query_times.size, 3), dtype=float)
        poses[:, 0] = np.interp(query_times, self.times, self.poses[:, 0])
        poses[:, 1] = np.interp(query_times, self.times, self.poses[:, 1])
        unwrapped_yaw = np.unwrap(self.poses[:, 2])
        poses[:, 2] = np.interp(query_times, self.times, unwrapped_yaw)
        return poses

    def preview(self, current_time: float, dt: float, horizon_steps: int) -> ReferencePreview:
        state_times = current_time + dt * np.arange(horizon_steps + 1)
        control_times = state_times[:-1]
        state_times = np.clip(state_times, self.times[0], self.times[-1])
        control_times = np.clip(control_times, self.times[0], self.times[-1])
        return ReferencePreview(
            poses=self._interpolate_pose(state_times),
            speeds=np.interp(control_times, self.times, self.speeds),
            yaw_rates=np.interp(control_times, self.times, self.yaw_rates),
        )


def wrapped_angle_error(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))

