"""Command-line entry point for the reproducible NMPC demos."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import MPCParameters, RobotParameters, SimulationParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .nmpc import NMPCController
from .robot import RobotDescription
from .simulation import run_closed_loop
from .trajectory import ReferenceTrajectory
from .live_visualization import LiveSimulationPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SERPUGA kinematic NMPC simulation.",
    )
    parser.add_argument(
        "--scenario",
        choices=("gap", "turn", "opposed"),
        default="gap",
        help="Synthetic scenario: standard gap, turn, or opposed-track start.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the real-time visualiser (for CI or remote shells).",
    )
    parser.add_argument(
        "--screenshot",
        "--output",
        dest="screenshot",
        type=Path,
        default=None,
        help="Optionally save a frame of the live visualiser as a PNG.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optionally export the 1x replay as an MP4 or GIF.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    parser.add_argument(
        "--no-zmp",
        action="store_true",
        help="Use the quasi-static CoM margin instead of the approximate ZMP.",
    )
    return parser


def run_demo(arguments: argparse.Namespace):
    robot_parameters = (
        RobotParameters.opposed_tracks()
        if arguments.scenario == "opposed"
        else RobotParameters()
    )
    mpc_parameters = MPCParameters(use_zmp=not arguments.no_zmp)
    simulation_parameters = SimulationParameters()
    if arguments.scenario == "opposed":
        initial_state = simulation_parameters.initial_state.copy()
        initial_state[3:5] = robot_parameters.nominal_configuration
        simulation_parameters = replace(
            simulation_parameters,
            initial_state=initial_state,
        )
    robot = RobotDescription(robot_parameters)
    model = KinematicModel(robot, mpc_parameters)

    if arguments.scenario in ("gap", "opposed"):
        corridor = StraightGapCorridor()
        trajectory = ReferenceTrajectory.straight(
            duration=simulation_parameters.duration + mpc_parameters.horizon_time,
            integration_dt=mpc_parameters.dt,
            speed=simulation_parameters.desired_speed,
        )
    else:
        corridor = StraightGapCorridor(
            open_width=4.0,
            gap_width=4.0,
            gap_start=100.0,
            gap_end=101.0,
        )
        simulation_parameters = replace(
            simulation_parameters,
            duration=8.0,
            stop_position=10.0,
        )
        trajectory = ReferenceTrajectory.gentle_turn(
            duration=simulation_parameters.duration + mpc_parameters.horizon_time,
            integration_dt=mpc_parameters.dt,
            speed=0.28,
        )

    controller = NMPCController(robot, model, corridor, mpc_parameters)
    log = run_closed_loop(
        controller=controller,
        model=model,
        robot=robot,
        corridor=corridor,
        trajectory=trajectory,
        mpc_parameters=mpc_parameters,
        simulation_parameters=simulation_parameters,
        verbose=not arguments.quiet,
    )
    print(json.dumps(log.summary(), indent=2, ensure_ascii=False))

    player: LiveSimulationPlayer | None = None
    if (
        not arguments.headless
        or arguments.screenshot is not None
        or arguments.video is not None
    ):
        player = LiveSimulationPlayer(
            log=log,
            robot=robot,
            corridor=corridor,
            mpc_parameters=mpc_parameters,
        )
    if arguments.screenshot is not None and player is not None:
        screenshot = player.save_frame(arguments.screenshot)
        print(f"Visualiser screenshot saved to {screenshot.resolve()}")
    if arguments.video is not None and player is not None:
        video = player.save_animation(arguments.video)
        print(f"Visualiser animation saved to {video.resolve()}")
    if not arguments.headless and player is not None:
        if not arguments.quiet:
            print("Opening real-time replay at 1x. Close the window to exit.")
        player.show()
    if player is not None:
        player.close()
    return log


def main() -> None:
    run_demo(build_parser().parse_args())


if __name__ == "__main__":
    main()
