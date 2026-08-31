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
        previous_control=np.zeros(3),
        previous_world_velocity=np.zeros(2),
    )
    assert solution.success
    assert solution.body_twist[0] > 0.20
    assert abs(solution.body_twist[1]) < 0.02
    assert abs(solution.body_twist[2]) < 0.02


def test_opposed_tracks_use_opposite_belt_speeds_for_forward_motion() -> None:
    parameters = MPCParameters(horizon_steps=4)
    robot_parameters = RobotParameters.opposed_tracks()
    robot = RobotDescription(robot_parameters)
    model = KinematicModel(robot, parameters)
    corridor = StraightGapCorridor(
        open_width=2.0,
        gap_width=2.0,
        gap_start=10.0,
        gap_end=11.0,
    )
    controller = NMPCController(robot, model, corridor, parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.25)
    state = np.zeros(5)
    state[3:5] = robot_parameters.nominal_configuration
    solution = controller.solve(
        state=state,
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
        previous_control=np.zeros(3),
        previous_world_velocity=np.zeros(2),
    )

    assert solution.success
    assert solution.control[0] > 0.0
    assert abs(solution.control[1]) < 0.02
    assert solution.body_twist[0] > 0.20
    assert solution.actuator_command[2] > 0.0
    assert solution.actuator_command[3] < 0.0
    projected_twist = model.body_twist(solution.control)
    np.testing.assert_allclose(
        solution.body_twist,
        projected_twist,
        atol=1.0e-9,
    )


def test_initial_yaw_offset_does_not_make_controller_infeasible() -> None:
    parameters = MPCParameters(horizon_steps=3)
    robot = RobotDescription(RobotParameters())
    model = KinematicModel(robot, parameters)
    corridor = StraightGapCorridor(
        open_width=2.0,
        gap_width=2.0,
        gap_start=10.0,
        gap_end=11.0,
    )
    controller = NMPCController(robot, model, corridor, parameters)
    trajectory = ReferenceTrajectory.straight(1.0, parameters.dt, 0.25)
    state = np.zeros(5)
    state[2] = np.deg2rad(90.0)

    solution = controller.solve(
        state=state,
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
        previous_control=np.zeros(3),
        previous_world_velocity=np.zeros(2),
    )

    assert solution.success
