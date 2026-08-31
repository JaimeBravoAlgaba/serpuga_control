import numpy as np

from serpuga_control import (
    KinematicModel,
    MPCParameters,
    RobotDescription,
    RobotParameters,
)
from serpuga_control.corridor import StraightGapCorridor
from serpuga_control.nmpc import NMPCController
from serpuga_control.trajectory import ReferenceTrajectory


def test_short_horizon_solution_tracks_forward_reference() -> None:
    parameters = MPCParameters(horizon_steps=4)
    robot = RobotDescription(RobotParameters())
    model = KinematicModel(robot, parameters)
    corridor = StraightGapCorridor(
        open_width=2.0,
        gap_width=2.0,
        gap_start=10.0,
        gap_end=11.0,
    )
    controller = NMPCController(robot, model, corridor, parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.25)
    solution = controller.solve(
        state=np.zeros(5),
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
        previous_control=np.zeros(4),
        previous_world_velocity=np.zeros(2),
    )
    assert solution.success
    assert solution.body_twist[0] > 0.20
    assert abs(solution.body_twist[1]) < 0.02
    assert abs(solution.body_twist[2]) < 0.02

