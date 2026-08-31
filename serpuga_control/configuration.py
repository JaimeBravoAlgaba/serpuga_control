"""User-facing simulation configuration and YAML profile storage.

The numerical dataclasses remain the contract used by the controller.  This
module is the single boundary exposed to users: it defines every editable
field, its unit, its UI grouping and its YAML representation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .config import MPCParameters, RobotParameters, SimulationParameters
from .corridor import StraightGapCorridor

CONFIG_VERSION = 1
FieldKind = Literal["float", "int", "bool", "optional_float"]


class ConfigurationError(ValueError):
    """Raised when a YAML profile or a form value is invalid."""


@dataclass(frozen=True)
class ParameterField:
    """Description shared by the GUI form and the YAML parser."""

    tab: str
    group: str
    section: str
    key: str
    label: str
    unit: str = ""
    kind: FieldKind = "float"

    @property
    def identifier(self) -> str:
        return f"{self.section}.{self.key}"


def _field(
    tab: str,
    group: str,
    section: str,
    key: str,
    label: str,
    unit: str = "",
    kind: FieldKind = "float",
) -> ParameterField:
    return ParameterField(tab, group, section, key, label, unit, kind)


FORM_FIELDS: tuple[ParameterField, ...] = (
    # Robot · geometry
    _field(
        "Robot", "Geometría", "robot", "pivot_1_x_m", "Pivote 1 · x desde centro", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "pivot_1_y_m", "Pivote 1 · y desde centro", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "pivot_2_x_m", "Pivote 2 · x desde centro", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "pivot_2_y_m", "Pivote 2 · y desde centro", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "track_1_offset_x_m", "Offset oruga 1 · x", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "track_1_offset_y_m", "Offset oruga 1 · y", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "track_2_offset_x_m", "Offset oruga 2 · x", "m"
    ),
    _field(
        "Robot", "Geometría", "robot", "track_2_offset_y_m", "Offset oruga 2 · y", "m"
    ),
    _field("Robot", "Geometría", "robot", "track_length_m", "Longitud de oruga", "m"),
    _field("Robot", "Geometría", "robot", "track_width_m", "Anchura de oruga", "m"),
    _field(
        "Robot", "Geometría", "robot", "connector_thickness_m", "Espesor del brazo", "m"
    ),
    # Robot · mass
    _field(
        "Robot",
        "Masa y centro de masas",
        "robot",
        "body_mass_kg",
        "Masa del cuerpo",
        "kg",
    ),
    _field(
        "Robot",
        "Masa y centro de masas",
        "robot",
        "track_mass_kg",
        "Masa de cada oruga",
        "kg",
    ),
    _field(
        "Robot",
        "Masa y centro de masas",
        "robot",
        "body_com_x_m",
        "CoM cuerpo · x",
        "m",
    ),
    _field(
        "Robot",
        "Masa y centro de masas",
        "robot",
        "body_com_y_m",
        "CoM cuerpo · y",
        "m",
    ),
    _field(
        "Robot",
        "Masa y centro de masas",
        "robot",
        "com_height_m",
        "Altura del CoM",
        "m",
    ),
    # Robot · articulation
    _field("Robot", "Articulaciones", "robot", "q1_min_deg", "q1 mínimo", "deg"),
    _field("Robot", "Articulaciones", "robot", "q1_max_deg", "q1 máximo", "deg"),
    _field("Robot", "Articulaciones", "robot", "q2_min_deg", "q2 mínimo", "deg"),
    _field("Robot", "Articulaciones", "robot", "q2_max_deg", "q2 máximo", "deg"),
    _field("Robot", "Articulaciones", "robot", "q1_nominal_deg", "q1 nominal", "deg"),
    _field("Robot", "Articulaciones", "robot", "q2_nominal_deg", "q2 nominal", "deg"),
    _field("Robot", "Articulaciones", "robot", "symmetry_1", "Acoplamiento q1"),
    _field("Robot", "Articulaciones", "robot", "symmetry_2", "Acoplamiento q2"),
    # Robot · actuators and contact slip
    _field(
        "Robot",
        "Actuadores",
        "robot",
        "track_speed_limit_mps",
        "Velocidad máxima de banda",
        "m/s",
    ),
    _field(
        "Robot",
        "Actuadores",
        "robot",
        "articulation_rate_limit_degps",
        "Velocidad articular máxima",
        "deg/s",
    ),
    _field(
        "Robot",
        "Actuadores",
        "robot",
        "track_acceleration_limit_mps2",
        "Aceleración máxima de banda",
        "m/s²",
    ),
    _field(
        "Robot",
        "Actuadores",
        "robot",
        "articulation_acceleration_limit_degps2",
        "Aceleración articular máxima",
        "deg/s²",
    ),
    _field(
        "Robot",
        "Deslizamiento de contacto",
        "robot",
        "longitudinal_slip_weight",
        "Peso longitudinal",
    ),
    _field(
        "Robot",
        "Deslizamiento de contacto",
        "robot",
        "lateral_slip_weight",
        "Peso lateral",
    ),
    # Scenario
    _field(
        "Escenario",
        "Corredor",
        "scenario",
        "open_width_m",
        "Anchura fuera del hueco",
        "m",
    ),
    _field(
        "Escenario", "Corredor", "scenario", "gap_width_m", "Anchura del hueco", "m"
    ),
    _field("Escenario", "Corredor", "scenario", "gap_start_m", "Inicio del hueco", "m"),
    _field("Escenario", "Corredor", "scenario", "gap_end_m", "Final del hueco", "m"),
    _field(
        "Escenario",
        "Corredor",
        "scenario",
        "transition_length_m",
        "Longitud de transición",
        "m",
    ),
    _field("Escenario", "Corredor", "scenario", "centre_y_m", "Centro lateral", "m"),
    # Simulation
    _field(
        "Simulación",
        "Tiempo y referencia",
        "simulation",
        "duration_s",
        "Duración máxima",
        "s",
    ),
    _field(
        "Simulación",
        "Tiempo y referencia",
        "simulation",
        "desired_speed_mps",
        "Velocidad lineal deseada",
        "m/s",
    ),
    _field(
        "Simulación",
        "Tiempo y referencia",
        "simulation",
        "desired_yaw_rate_rps",
        "Velocidad angular deseada",
        "rad/s",
    ),
    _field(
        "Simulación",
        "Tiempo y referencia",
        "simulation",
        "stop_x_m",
        "Detener al alcanzar x",
        "m",
        "optional_float",
    ),
    _field(
        "Simulación", "Estado inicial", "simulation", "initial_x_m", "x inicial", "m"
    ),
    _field(
        "Simulación", "Estado inicial", "simulation", "initial_y_m", "y inicial", "m"
    ),
    _field(
        "Simulación",
        "Estado inicial",
        "simulation",
        "initial_yaw_deg",
        "Yaw inicial",
        "deg",
    ),
    _field(
        "Simulación",
        "Estado inicial",
        "simulation",
        "initial_q1_deg",
        "q1 inicial",
        "deg",
    ),
    _field(
        "Simulación",
        "Estado inicial",
        "simulation",
        "initial_q2_deg",
        "q2 inicial",
        "deg",
    ),
    _field(
        "Simulación",
        "Reproducibilidad",
        "simulation",
        "random_seed",
        "Semilla aleatoria",
        kind="int",
    ),
    # MPC · horizon and costs
    _field("MPC", "Horizonte", "mpc", "sample_time_s", "Periodo de control", "s"),
    _field(
        "MPC", "Horizonte", "mpc", "horizon_steps", "Pasos del horizonte", kind="int"
    ),
    _field("MPC", "Costes de seguimiento", "mpc", "position_weight", "Posición"),
    _field("MPC", "Costes de seguimiento", "mpc", "heading_weight", "Orientación"),
    _field(
        "MPC", "Costes de seguimiento", "mpc", "velocity_weight", "Velocidad lineal"
    ),
    _field(
        "MPC", "Costes de seguimiento", "mpc", "yaw_rate_weight", "Velocidad angular"
    ),
    _field(
        "MPC",
        "Costes de seguimiento",
        "mpc",
        "terminal_position_weight",
        "Posición terminal",
    ),
    _field(
        "MPC",
        "Costes de seguimiento",
        "mpc",
        "terminal_heading_weight",
        "Orientación terminal",
    ),
    _field(
        "MPC", "Costes de contacto y actuación", "mpc", "slip_weight", "Deslizamiento"
    ),
    _field("MPC", "Costes de contacto y actuación", "mpc", "scrub_weight", "Scrubbing"),
    _field(
        "MPC",
        "Costes de contacto y actuación",
        "mpc",
        "articulation_rate_weight",
        "Velocidad articular",
    ),
    _field(
        "MPC",
        "Costes de contacto y actuación",
        "mpc",
        "track_effort_weight",
        "Esfuerzo de bandas",
    ),
    _field(
        "MPC",
        "Costes de contacto y actuación",
        "mpc",
        "input_rate_weight",
        "Variación de comandos",
    ),
    _field(
        "MPC", "Costes de forma y estabilidad", "mpc", "symmetry_weight", "Simetría"
    ),
    _field(
        "MPC",
        "Costes de forma y estabilidad",
        "mpc",
        "track_alignment_weight",
        "Alineamiento de orugas",
    ),
    _field(
        "MPC", "Costes de forma y estabilidad", "mpc", "stability_weight", "Estabilidad"
    ),
    _field(
        "MPC",
        "Costes de forma y estabilidad",
        "mpc",
        "nominal_configuration_weight",
        "Configuración nominal",
    ),
    # MPC · constraints
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "clearance_margin_m",
        "Margen contra paredes",
        "m",
    ),
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "minimum_stability_margin_m",
        "Margen mínimo de estabilidad",
        "m",
    ),
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "target_stability_margin_m",
        "Margen objetivo de estabilidad",
        "m",
    ),
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "maximum_heading_error_deg",
        "Umbral blando de orientación",
        "deg",
    ),
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "body_speed_limit_mps",
        "Velocidad máxima del cuerpo",
        "m/s",
    ),
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "body_yaw_rate_limit_rps",
        "Yaw rate máximo del cuerpo",
        "rad/s",
    ),
    _field(
        "MPC",
        "Restricciones",
        "mpc",
        "maximum_lateral_slip_mps",
        "Slip lateral máximo",
        "m/s",
    ),
    # MPC · model and solver
    _field(
        "MPC",
        "Modelo de estabilidad",
        "mpc",
        "use_zmp",
        "Usar ZMP aproximado",
        kind="bool",
    ),
    _field("MPC", "Modelo de estabilidad", "mpc", "gravity_mps2", "Gravedad", "m/s²"),
    _field("MPC", "Solver", "mpc", "regularisation", "Regularización"),
    _field("MPC", "Solver", "mpc", "smooth_epsilon", "Épsilon de suavizado"),
    _field(
        "MPC",
        "Solver",
        "mpc",
        "ipopt_max_iterations",
        "Iteraciones máximas IPOPT",
        kind="int",
    ),
    _field("MPC", "Solver", "mpc", "ipopt_tolerance", "Tolerancia IPOPT"),
)


@dataclass(frozen=True)
class ApplicationConfiguration:
    """Complete set of parameters needed by one simulation run."""

    robot: RobotParameters
    corridor: StraightGapCorridor
    mpc: MPCParameters
    simulation: SimulationParameters

    def validate(self) -> None:
        r = self.robot
        c = self.corridor
        p = self.mpc
        s = self.simulation

        positive = {
            "robot.track_length": r.track_length,
            "robot.track_width": r.track_width,
            "robot.connector_thickness": r.connector_thickness,
            "robot.body_mass": r.body_mass,
            "robot.track_mass": r.track_mass,
            "robot.com_height": r.com_height,
            "robot.track_speed_limit": r.track_speed_limit,
            "robot.articulation_rate_limit": r.articulation_rate_limit,
            "robot.track_acceleration_limit": r.track_acceleration_limit,
            "robot.articulation_acceleration_limit": r.articulation_acceleration_limit,
            "scenario.open_width": c.open_width,
            "scenario.gap_width": c.gap_width,
            "scenario.transition_length": c.transition_length,
            "simulation.duration": s.duration,
            "mpc.dt": p.dt,
            "mpc.body_speed_limit": p.body_speed_limit,
            "mpc.body_yaw_rate_limit": p.body_yaw_rate_limit,
            "mpc.gravity": p.gravity,
            "mpc.maximum_heading_error": p.maximum_heading_error,
            "mpc.regularisation": p.regularisation,
            "mpc.smooth_epsilon": p.smooth_epsilon,
            "mpc.ipopt_tolerance": p.ipopt_tolerance,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ConfigurationError(f"{name} must be greater than zero")
        if c.gap_width > c.open_width:
            raise ConfigurationError("scenario.gap_width must not exceed open_width")
        if c.gap_end <= c.gap_start:
            raise ConfigurationError("scenario.gap_end must be greater than gap_start")
        if np.linalg.norm(r.pivot_positions[1] - r.pivot_positions[0]) <= 0.0:
            raise ConfigurationError("The two robot pivots must be different")
        if p.horizon_steps < 1 or p.ipopt_max_iterations < 1:
            raise ConfigurationError(
                "MPC horizon and solver iterations must be positive"
            )
        if p.clearance_margin < 0.0 or p.minimum_stability_margin < 0.0:
            raise ConfigurationError(
                "Clearance and stability margins cannot be negative"
            )
        if p.maximum_lateral_slip < 0.0:
            raise ConfigurationError("Maximum lateral slip cannot be negative")
        if p.target_stability_margin < p.minimum_stability_margin:
            raise ConfigurationError(
                "Target stability margin must be at least the minimum margin"
            )
        if c.gap_width <= 2.0 * p.clearance_margin:
            raise ConfigurationError(
                "Gap width must exceed twice the configured wall clearance"
            )
        if np.any(s.initial_state[3:5] < r.q_min) or np.any(
            s.initial_state[3:5] > r.q_max
        ):
            raise ConfigurationError("Initial q1/q2 must lie inside their joint limits")
        cost_names = (
            "position_weight",
            "heading_weight",
            "velocity_weight",
            "yaw_rate_weight",
            "slip_weight",
            "scrub_weight",
            "articulation_rate_weight",
            "track_effort_weight",
            "input_rate_weight",
            "symmetry_weight",
            "track_alignment_weight",
            "stability_weight",
            "nominal_configuration_weight",
            "terminal_position_weight",
            "terminal_heading_weight",
        )
        for name in cost_names:
            if getattr(p, name) < 0.0:
                raise ConfigurationError(f"mpc.{name} cannot be negative")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ApplicationConfiguration:
        version = int(mapping.get("version", CONFIG_VERSION))
        if version != CONFIG_VERSION:
            raise ConfigurationError(
                f"Unsupported configuration version {version}; expected {CONFIG_VERSION}"
            )

        sections: dict[str, Mapping[str, Any]] = {}
        for name in ("robot", "scenario", "simulation", "mpc"):
            section = mapping.get(name)
            if not isinstance(section, Mapping):
                raise ConfigurationError(f"Missing YAML section: {name}")
            sections[name] = section

        def required(section: str, key: str) -> Any:
            if key not in sections[section]:
                raise ConfigurationError(f"Missing YAML value: {section}.{key}")
            return sections[section][key]

        def number(section: str, key: str) -> float:
            value = required(section, key)
            if isinstance(value, bool):
                raise ConfigurationError(f"{section}.{key} must be numeric")
            try:
                result = float(value)
            except (TypeError, ValueError) as error:
                raise ConfigurationError(f"{section}.{key} must be numeric") from error
            if not np.isfinite(result):
                raise ConfigurationError(f"{section}.{key} must be finite")
            return result

        def integer(section: str, key: str) -> int:
            value = number(section, key)
            if not value.is_integer():
                raise ConfigurationError(f"{section}.{key} must be an integer")
            return int(value)

        def boolean(section: str, key: str) -> bool:
            value = required(section, key)
            if not isinstance(value, bool):
                raise ConfigurationError(f"{section}.{key} must be true or false")
            return value

        def optional_number(section: str, key: str) -> float | None:
            value = required(section, key)
            if value is None or value == "":
                return None
            return number(section, key)

        deg = np.deg2rad
        robot = RobotParameters(
            pivot_positions=np.array(
                [
                    [number("robot", "pivot_1_x_m"), number("robot", "pivot_1_y_m")],
                    [number("robot", "pivot_2_x_m"), number("robot", "pivot_2_y_m")],
                ],
                dtype=float,
            ),
            track_center_offsets=np.array(
                [
                    [
                        number("robot", "track_1_offset_x_m"),
                        number("robot", "track_1_offset_y_m"),
                    ],
                    [
                        number("robot", "track_2_offset_x_m"),
                        number("robot", "track_2_offset_y_m"),
                    ],
                ],
                dtype=float,
            ),
            track_length=number("robot", "track_length_m"),
            track_width=number("robot", "track_width_m"),
            connector_thickness=number("robot", "connector_thickness_m"),
            body_mass=number("robot", "body_mass_kg"),
            track_mass=number("robot", "track_mass_kg"),
            body_com=np.array(
                [number("robot", "body_com_x_m"), number("robot", "body_com_y_m")],
                dtype=float,
            ),
            com_height=number("robot", "com_height_m"),
            q_min=deg(
                np.array([number("robot", "q1_min_deg"), number("robot", "q2_min_deg")])
            ),
            q_max=deg(
                np.array([number("robot", "q1_max_deg"), number("robot", "q2_max_deg")])
            ),
            nominal_configuration=deg(
                np.array(
                    [
                        number("robot", "q1_nominal_deg"),
                        number("robot", "q2_nominal_deg"),
                    ]
                )
            ),
            symmetry_coupling=np.array(
                [number("robot", "symmetry_1"), number("robot", "symmetry_2")],
                dtype=float,
            ),
            track_speed_limit=number("robot", "track_speed_limit_mps"),
            articulation_rate_limit=float(
                deg(number("robot", "articulation_rate_limit_degps"))
            ),
            track_acceleration_limit=number("robot", "track_acceleration_limit_mps2"),
            articulation_acceleration_limit=float(
                deg(number("robot", "articulation_acceleration_limit_degps2"))
            ),
            longitudinal_slip_weight=number("robot", "longitudinal_slip_weight"),
            lateral_slip_weight=number("robot", "lateral_slip_weight"),
        )

        corridor = StraightGapCorridor(
            open_width=number("scenario", "open_width_m"),
            gap_width=number("scenario", "gap_width_m"),
            gap_start=number("scenario", "gap_start_m"),
            gap_end=number("scenario", "gap_end_m"),
            transition_length=number("scenario", "transition_length_m"),
            centre_y=number("scenario", "centre_y_m"),
        )

        p = lambda key: number("mpc", key)
        mpc = MPCParameters(
            dt=p("sample_time_s"),
            horizon_steps=integer("mpc", "horizon_steps"),
            position_weight=p("position_weight"),
            heading_weight=p("heading_weight"),
            velocity_weight=p("velocity_weight"),
            yaw_rate_weight=p("yaw_rate_weight"),
            slip_weight=p("slip_weight"),
            scrub_weight=p("scrub_weight"),
            articulation_rate_weight=p("articulation_rate_weight"),
            track_effort_weight=p("track_effort_weight"),
            input_rate_weight=p("input_rate_weight"),
            symmetry_weight=p("symmetry_weight"),
            track_alignment_weight=p("track_alignment_weight"),
            stability_weight=p("stability_weight"),
            nominal_configuration_weight=p("nominal_configuration_weight"),
            terminal_position_weight=p("terminal_position_weight"),
            terminal_heading_weight=p("terminal_heading_weight"),
            clearance_margin=p("clearance_margin_m"),
            minimum_stability_margin=p("minimum_stability_margin_m"),
            target_stability_margin=p("target_stability_margin_m"),
            maximum_heading_error=float(deg(p("maximum_heading_error_deg"))),
            body_speed_limit=p("body_speed_limit_mps"),
            body_yaw_rate_limit=p("body_yaw_rate_limit_rps"),
            maximum_lateral_slip=p("maximum_lateral_slip_mps"),
            use_zmp=boolean("mpc", "use_zmp"),
            gravity=p("gravity_mps2"),
            regularisation=p("regularisation"),
            smooth_epsilon=p("smooth_epsilon"),
            ipopt_max_iterations=integer("mpc", "ipopt_max_iterations"),
            ipopt_tolerance=p("ipopt_tolerance"),
        )

        initial_state = np.array(
            [
                number("simulation", "initial_x_m"),
                number("simulation", "initial_y_m"),
                deg(number("simulation", "initial_yaw_deg")),
                deg(number("simulation", "initial_q1_deg")),
                deg(number("simulation", "initial_q2_deg")),
            ],
            dtype=float,
        )
        simulation = SimulationParameters(
            duration=number("simulation", "duration_s"),
            desired_speed=number("simulation", "desired_speed_mps"),
            desired_yaw_rate=number("simulation", "desired_yaw_rate_rps"),
            initial_state=initial_state,
            stop_position=optional_number("simulation", "stop_x_m"),
            random_seed=integer("simulation", "random_seed"),
        )

        configuration = cls(
            robot=robot, corridor=corridor, mpc=mpc, simulation=simulation
        )
        configuration.validate()
        return configuration

    def to_mapping(self) -> dict[str, Any]:
        r, c, p, s = self.robot, self.corridor, self.mpc, self.simulation
        degrees = np.rad2deg
        return {
            "version": CONFIG_VERSION,
            "robot": {
                "pivot_1_x_m": float(r.pivot_positions[0, 0]),
                "pivot_1_y_m": float(r.pivot_positions[0, 1]),
                "pivot_2_x_m": float(r.pivot_positions[1, 0]),
                "pivot_2_y_m": float(r.pivot_positions[1, 1]),
                "track_1_offset_x_m": float(r.track_center_offsets[0, 0]),
                "track_1_offset_y_m": float(r.track_center_offsets[0, 1]),
                "track_2_offset_x_m": float(r.track_center_offsets[1, 0]),
                "track_2_offset_y_m": float(r.track_center_offsets[1, 1]),
                "track_length_m": float(r.track_length),
                "track_width_m": float(r.track_width),
                "connector_thickness_m": float(r.connector_thickness),
                "body_mass_kg": float(r.body_mass),
                "track_mass_kg": float(r.track_mass),
                "body_com_x_m": float(r.body_com[0]),
                "body_com_y_m": float(r.body_com[1]),
                "com_height_m": float(r.com_height),
                "q1_min_deg": float(degrees(r.q_min[0])),
                "q1_max_deg": float(degrees(r.q_max[0])),
                "q2_min_deg": float(degrees(r.q_min[1])),
                "q2_max_deg": float(degrees(r.q_max[1])),
                "q1_nominal_deg": float(degrees(r.nominal_configuration[0])),
                "q2_nominal_deg": float(degrees(r.nominal_configuration[1])),
                "symmetry_1": float(r.symmetry_coupling[0]),
                "symmetry_2": float(r.symmetry_coupling[1]),
                "track_speed_limit_mps": float(r.track_speed_limit),
                "articulation_rate_limit_degps": float(
                    degrees(r.articulation_rate_limit)
                ),
                "track_acceleration_limit_mps2": float(r.track_acceleration_limit),
                "articulation_acceleration_limit_degps2": float(
                    degrees(r.articulation_acceleration_limit)
                ),
                "longitudinal_slip_weight": float(r.longitudinal_slip_weight),
                "lateral_slip_weight": float(r.lateral_slip_weight),
            },
            "scenario": {
                "open_width_m": float(c.open_width),
                "gap_width_m": float(c.gap_width),
                "gap_start_m": float(c.gap_start),
                "gap_end_m": float(c.gap_end),
                "transition_length_m": float(c.transition_length),
                "centre_y_m": float(c.centre_y),
            },
            "simulation": {
                "duration_s": float(s.duration),
                "desired_speed_mps": float(s.desired_speed),
                "desired_yaw_rate_rps": float(s.desired_yaw_rate),
                "stop_x_m": None if s.stop_position is None else float(s.stop_position),
                "initial_x_m": float(s.initial_state[0]),
                "initial_y_m": float(s.initial_state[1]),
                "initial_yaw_deg": float(degrees(s.initial_state[2])),
                "initial_q1_deg": float(degrees(s.initial_state[3])),
                "initial_q2_deg": float(degrees(s.initial_state[4])),
                "random_seed": int(s.random_seed),
            },
            "mpc": {
                "sample_time_s": float(p.dt),
                "horizon_steps": int(p.horizon_steps),
                "position_weight": float(p.position_weight),
                "heading_weight": float(p.heading_weight),
                "velocity_weight": float(p.velocity_weight),
                "yaw_rate_weight": float(p.yaw_rate_weight),
                "slip_weight": float(p.slip_weight),
                "scrub_weight": float(p.scrub_weight),
                "articulation_rate_weight": float(p.articulation_rate_weight),
                "track_effort_weight": float(p.track_effort_weight),
                "input_rate_weight": float(p.input_rate_weight),
                "symmetry_weight": float(p.symmetry_weight),
                "track_alignment_weight": float(p.track_alignment_weight),
                "stability_weight": float(p.stability_weight),
                "nominal_configuration_weight": float(p.nominal_configuration_weight),
                "terminal_position_weight": float(p.terminal_position_weight),
                "terminal_heading_weight": float(p.terminal_heading_weight),
                "clearance_margin_m": float(p.clearance_margin),
                "minimum_stability_margin_m": float(p.minimum_stability_margin),
                "target_stability_margin_m": float(p.target_stability_margin),
                "maximum_heading_error_deg": float(degrees(p.maximum_heading_error)),
                "body_speed_limit_mps": float(p.body_speed_limit),
                "body_yaw_rate_limit_rps": float(p.body_yaw_rate_limit),
                "maximum_lateral_slip_mps": float(p.maximum_lateral_slip),
                "use_zmp": bool(p.use_zmp),
                "gravity_mps2": float(p.gravity),
                "regularisation": float(p.regularisation),
                "smooth_epsilon": float(p.smooth_epsilon),
                "ipopt_max_iterations": int(p.ipopt_max_iterations),
                "ipopt_tolerance": float(p.ipopt_tolerance),
            },
        }


def configuration_to_form_values(
    configuration: ApplicationConfiguration,
) -> dict[str, str | bool]:
    mapping = configuration.to_mapping()
    values: dict[str, str | bool] = {}
    for spec in FORM_FIELDS:
        value = mapping[spec.section][spec.key]
        if spec.kind == "bool":
            values[spec.identifier] = bool(value)
        elif value is None:
            values[spec.identifier] = ""
        elif spec.kind == "int":
            values[spec.identifier] = str(int(value))
        else:
            values[spec.identifier] = f"{float(value):.12g}"
    return values


def configuration_from_form_values(
    values: Mapping[str, str | bool],
) -> ApplicationConfiguration:
    mapping: dict[str, Any] = {
        "version": CONFIG_VERSION,
        "robot": {},
        "scenario": {},
        "simulation": {},
        "mpc": {},
    }
    for spec in FORM_FIELDS:
        if spec.identifier not in values:
            raise ConfigurationError(f"Missing form value: {spec.label}")
        raw = values[spec.identifier]
        try:
            if spec.kind == "bool":
                if isinstance(raw, bool):
                    parsed: Any = raw
                elif str(raw).strip().lower() in ("true", "1", "yes", "on"):
                    parsed = True
                elif str(raw).strip().lower() in ("false", "0", "no", "off"):
                    parsed = False
                else:
                    raise ValueError("expected a boolean")
            elif spec.kind == "optional_float" and str(raw).strip() == "":
                parsed = None
            elif spec.kind == "int":
                parsed = int(str(raw).strip())
            else:
                parsed = float(str(raw).strip())
        except (TypeError, ValueError) as error:
            unit = f" [{spec.unit}]" if spec.unit else ""
            raise ConfigurationError(
                f"Invalid value for {spec.label}{unit}: {raw!r}"
            ) from error
        mapping[spec.section][spec.key] = parsed
    try:
        return ApplicationConfiguration.from_mapping(mapping)
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigurationError(str(error)) from error


class ConfigurationStore:
    """List, load and save editable YAML profiles in one directory."""

    def __init__(self, directory: str | Path = "configs") -> None:
        self.directory = Path(directory)

    def list_profiles(self) -> list[str]:
        if not self.directory.exists():
            return []
        names = {
            path.stem
            for pattern in ("*.yaml", "*.yml")
            for path in self.directory.glob(pattern)
        }
        return sorted(names, key=lambda name: (name != "default", name.casefold()))

    def _resolve(self, name_or_path: str | Path) -> Path:
        candidate = Path(name_or_path)
        if candidate.is_absolute() or candidate.parent != Path("."):
            return candidate
        if candidate.suffix.lower() in (".yaml", ".yml"):
            return self.directory / candidate
        yaml_path = self.directory / f"{candidate.name}.yaml"
        yml_path = self.directory / f"{candidate.name}.yml"
        return yml_path if yml_path.exists() and not yaml_path.exists() else yaml_path

    def load(self, name_or_path: str | Path) -> ApplicationConfiguration:
        path = self._resolve(name_or_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ConfigurationError(f"Invalid YAML in {path}: {error}") from error
        if not isinstance(data, Mapping):
            raise ConfigurationError(
                f"The configuration root must be a mapping: {path}"
            )
        try:
            return ApplicationConfiguration.from_mapping(data)
        except ConfigurationError:
            raise
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"Invalid values in {path}: {error}") from error

    @staticmethod
    def safe_name(display_name: str) -> str:
        ascii_name = (
            unicodedata.normalize("NFKD", display_name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        clean = re.sub(r"[^A-Za-z0-9_-]+", "-", ascii_name.strip()).strip("-_")
        if not clean:
            raise ConfigurationError("The configuration name is empty")
        return clean.lower()

    def save(
        self,
        name: str,
        configuration: ApplicationConfiguration,
    ) -> Path:
        safe_name = self.safe_name(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{safe_name}.yaml"
        content = yaml.safe_dump(
            configuration.to_mapping(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        path.write_text(content, encoding="utf-8")
        return path
