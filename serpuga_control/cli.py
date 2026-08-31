"""Command-line entry point for the interactive SERPUGA application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .configuration import ConfigurationError, ConfigurationStore
from .live_visualization import LiveSimulationPlayer
from .runtime import build_runtime
from .simulation import run_closed_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and run the SERPUGA kinematic NMPC simulation.",
    )
    parser.add_argument(
        "--config",
        default="default",
        metavar="NAME_OR_PATH",
        help="Initial YAML profile name or path (default: default).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory listed by the profile selector (default: ./configs).",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available YAML profile names and exit.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the selected profile to completion without opening the GUI.",
    )
    parser.add_argument(
        "--screenshot",
        "--output",
        dest="screenshot",
        type=Path,
        default=None,
        help="Batch-run the profile and save one replay frame as PNG.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Batch-run the profile and export its replay as MP4 or GIF.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages."
    )
    return parser


def run_profile(
    store: ConfigurationStore,
    profile: str,
    *,
    quiet: bool = False,
    screenshot: Path | None = None,
    video: Path | None = None,
):
    configuration = store.load(profile)
    runtime = build_runtime(configuration)
    log = run_closed_loop(
        controller=runtime.controller,
        model=runtime.model,
        robot=runtime.robot,
        corridor=configuration.corridor,
        trajectory=runtime.trajectory,
        mpc_parameters=configuration.mpc,
        simulation_parameters=configuration.simulation,
        verbose=not quiet,
    )
    print(json.dumps(log.summary(), indent=2, ensure_ascii=False))

    if screenshot is not None or video is not None:
        player = LiveSimulationPlayer(
            log=log,
            robot=runtime.robot,
            corridor=configuration.corridor,
            mpc_parameters=configuration.mpc,
        )
        if screenshot is not None:
            output = player.save_frame(screenshot)
            print(f"Visualiser screenshot saved to {output.resolve()}")
        if video is not None:
            output = player.save_animation(video)
            print(f"Visualiser animation saved to {output.resolve()}")
        player.close()
    return log


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    store = ConfigurationStore(arguments.config_dir)

    if arguments.list_configs:
        for name in store.list_profiles():
            print(name)
        return

    try:
        if (
            arguments.headless
            or arguments.screenshot is not None
            or arguments.video is not None
        ):
            run_profile(
                store,
                arguments.config,
                quiet=arguments.quiet,
                screenshot=arguments.screenshot,
                video=arguments.video,
            )
        else:
            from .app import launch_application

            launch_application(store=store, initial_profile=arguments.config)
    except ConfigurationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
