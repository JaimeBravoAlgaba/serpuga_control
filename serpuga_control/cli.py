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
from .visualization import plot_simulation_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SERPUGA kinematic NMPC simulation.",
    )
    parser.add_argument(
        "--scenario",
        choices=("gap", "turn"),
        default="gap",
        help="Synthetic scenario to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/serpuga_visualizer.png"),
        help="Destination of the visualiser dashboard.",
    )
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib window.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    parser.add_argument(
        "--no-zmp",
        action="store_true",
        help="Use the quasi-static CoM margin instead of the approximate ZMP.",
    )
    return parser


def run_demo(arguments: argparse.Namespace):
    robot_parameters = RobotParameters()
    mpc_parameters = MPCParameters(use_zmp=not arguments.no_zmp)
    simulation_parameters = SimulationParameters()
    robot = RobotDescription(robot_parameters)
    model = KinematicModel(robot, mpc_parameters)

    if arguments.scenario == "gap":
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
    output = plot_simulation_dashboard(
        log=log,
        robot=robot,
        corridor=corridor,
        mpc_parameters=mpc_parameters,
        output_path=arguments.output,
        show=arguments.show,
    )
    print(json.dumps(log.summary(), indent=2, ensure_ascii=False))
    print(f"Visualiser saved to {output.resolve()}")
    return log


def main() -> None:
    run_demo(build_parser().parse_args())


if __name__ == "__main__":
    main()

