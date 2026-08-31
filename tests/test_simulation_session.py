from serpuga_control import (
    KinematicModel,
    MPCParameters,
    RobotDescription,
    RobotParameters,
)
from serpuga_control.config import SimulationParameters
from serpuga_control.corridor import StraightGapCorridor
from serpuga_control.nmpc import NMPCController
from serpuga_control.simulation import ClosedLoopSession
from serpuga_control.trajectory import ReferenceTrajectory


def test_closed_loop_session_advances_one_mpc_step_at_a_time() -> None:
    parameters = MPCParameters(horizon_steps=4)
    simulation = SimulationParameters(duration=0.30, stop_position=None)
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
    session = ClosedLoopSession(
        controller=NMPCController(robot, model, corridor, parameters),
        model=model,
        robot=robot,
        corridor=corridor,
        trajectory=trajectory,
        mpc_parameters=parameters,
        simulation_parameters=simulation,
    )

    assert session.times == []
    assert session.step()
    first_log = session.to_log()
    assert first_log.times.shape == (1,)
    assert first_log.states.shape == (2, 5)
    assert first_log.states[-1, 0] > 0.0
    assert not session.finished

    assert session.step()
    assert session.finished
    assert session.completed
    assert session.to_log().states.shape == (3, 5)
