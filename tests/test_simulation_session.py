import numpy as np

from serpuga_control import (
    KinematicModel,
    MPCParameters,
    RobotDescription,
    RobotParameters,
)
from serpuga_control.config import SimulationParameters
from serpuga_control.corridor import StraightGapCorridor
from serpuga_control.nmpc import NMPCController
from serpuga_control.simulation import ClosedLoopSession, TeleoperationCommand
from serpuga_control.trajectory import ReferenceTrajectory


def make_session(parameters: MPCParameters, simulation: SimulationParameters) -> ClosedLoopSession:
    robot = RobotDescription(RobotParameters())
    model = KinematicModel(robot, parameters)
    corridor = StraightGapCorridor(
        open_width=2.0,
        gap_width=2.0,
        gap_start=10.0,
        gap_end=11.0,
    )
    trajectory = ReferenceTrajectory.constant_twist(
        duration=1.0,
        integration_dt=parameters.dt,
        speed=0.25,
        yaw_rate=0.0,
    )
    return ClosedLoopSession(
        controller=NMPCController(robot, model, corridor, parameters),
        model=model,
        robot=robot,
        corridor=corridor,
        trajectory=trajectory,
        mpc_parameters=parameters,
        simulation_parameters=simulation,
    )


def test_closed_loop_session_advances_one_mpc_step_at_a_time() -> None:
    parameters = MPCParameters(horizon_steps=4)
    simulation = SimulationParameters(duration=0.30, stop_position=None)
    session = make_session(parameters, simulation)

    assert session.times == []
    assert session.step()
    first_log = session.to_log()
    assert first_log.times.shape == (1,)
    assert first_log.controls.shape == (1, 4)
    assert first_log.states.shape == (2, 5)
    assert first_log.states[-1, 0] > 0.0
    np.testing.assert_allclose(first_log.controls, first_log.actuator_commands)
    assert not session.finished

    assert session.step()
    assert session.finished
    assert session.completed
    assert session.to_log().states.shape == (3, 5)


def test_closed_loop_session_can_step_from_legacy_manual_teleoperation() -> None:
    parameters = MPCParameters(horizon_steps=4)
    simulation = SimulationParameters(duration=0.30, stop_position=None)
    session = make_session(parameters, simulation)
    command = TeleoperationCommand(enabled=True, body_twist=[9.0, 3.0, 0.3])

    assert session.step(command)

    log = session.to_log()
    assert log.control_modes == ["manual"]
    assert log.solver_statuses == ["Teleoperation"]
    assert log.solve_times[0] == 0.0
    assert log.controls.shape == (1, 4)
    np.testing.assert_allclose(log.controls, log.actuator_commands)
    assert np.max(np.abs(log.controls[0, 2:4])) <= parameters.track_speed_limit
    # Legacy manual mode keeps the current articulation and allocates only belts.
    np.testing.assert_allclose(log.states[-1, 3:5], log.states[0, 3:5])
    assert log.predicted_states[-1].shape == (parameters.horizon_steps + 1, 5)

    assert session.step(command, stop_when_complete=False)
    assert not session.finished
