"""Construction of numerical components from one application configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configuration import ApplicationConfiguration
from .kinematics import KinematicModel
from .nmpc import NMPCController
from .robot import RobotDescription
from .trajectory import ReferenceTrajectory


def build_trajectory(configuration: ApplicationConfiguration) -> ReferenceTrajectory:
    p = configuration.mpc
    s = configuration.simulation
    initial_reference_pose = np.array(
        [s.initial_state[0], s.initial_state[1], 0.0],
        dtype=float,
    )
    return ReferenceTrajectory.constant_twist(
        duration=s.duration + p.horizon_time,
        integration_dt=p.dt,
        speed=s.desired_speed,
        yaw_rate=s.desired_yaw_rate,
        initial_pose=initial_reference_pose,
    )


@dataclass
class SimulationRuntime:
    """Controller and plant objects belonging to one immutable run."""

    configuration: ApplicationConfiguration
    robot: RobotDescription
    model: KinematicModel
    trajectory: ReferenceTrajectory
    controller: NMPCController


def build_runtime(configuration: ApplicationConfiguration) -> SimulationRuntime:
    configuration.validate()
    robot = RobotDescription(configuration.robot)
    model = KinematicModel(robot, configuration.mpc)
    trajectory = build_trajectory(configuration)
    controller = NMPCController(
        robot=robot,
        model=model,
        corridor=configuration.corridor,
        parameters=configuration.mpc,
    )
    return SimulationRuntime(
        configuration=configuration,
        robot=robot,
        model=model,
        trajectory=trajectory,
        controller=controller,
    )
