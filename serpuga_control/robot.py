"""Parametric geometric description of the two-track robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .config import RobotParameters
from .math_utils import (
    as_column,
    is_symbolic,
    rotation_2d,
    smooth_abs,
    smooth_maximum,
    smooth_minimum,
)


@dataclass
class RobotDescription:
    """Geometry and mass properties used by control and visualisation."""

    parameters: RobotParameters

    @staticmethod
    def parallelism_residual(q: Any) -> Any:
        """Return zero for parallel or antiparallel track axes."""

        difference = q[0] - q[1]
        return ca.sin(difference) if is_symbolic(difference) else np.sin(difference)

    def track_corners_local(self) -> np.ndarray:
        half_length = 0.5 * self.parameters.track_length
        half_width = 0.5 * self.parameters.track_width
        return np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ],
            dtype=float,
        )

    def track_center_body(self, q_i: Any, track_index: int) -> Any:
        pivot = self.parameters.pivot_positions[track_index]
        offset = self.parameters.track_center_offsets[track_index]
        if is_symbolic(q_i):
            return as_column(pivot) + rotation_2d(q_i) @ as_column(offset)
        return pivot + rotation_2d(q_i) @ offset

    def track_center_derivative(self, q_i: Any, track_index: int) -> Any:
        offset = self.parameters.track_center_offsets[track_index]
        if is_symbolic(q_i):
            rotated = rotation_2d(q_i) @ as_column(offset)
            return ca.vertcat(-rotated[1], rotated[0])
        rotated = rotation_2d(q_i) @ offset
        return np.array([-rotated[1], rotated[0]], dtype=float)

    def track_vertices_body(self, q_i: Any, track_index: int) -> list[Any]:
        centre = self.track_center_body(q_i, track_index)
        rotation = rotation_2d(q_i)
        vertices: list[Any] = []
        for corner in self.track_corners_local():
            if is_symbolic(q_i):
                vertices.append(centre + rotation @ as_column(corner))
            else:
                vertices.append(centre + rotation @ corner)
        return vertices

    def connector_vertices_body(self) -> np.ndarray:
        """Return the four corners of the rigid bar joining both pivots."""

        first, second = self.parameters.pivot_positions
        direction = second - first
        length = np.linalg.norm(direction)
        if length <= 0.0:
            raise ValueError("The two pivot positions must be distinct")
        tangent = direction / length
        normal = np.array([-tangent[1], tangent[0]])
        half_thickness = 0.5 * self.parameters.connector_thickness
        return np.array(
            [
                first + half_thickness * normal,
                second + half_thickness * normal,
                second - half_thickness * normal,
                first - half_thickness * normal,
            ],
            dtype=float,
        )

    def track_vertices_world(self, state: Any, track_index: int) -> list[Any]:
        position = state[0:2]
        yaw = state[2]
        q_i = state[3 + track_index]
        body_rotation = rotation_2d(yaw)
        vertices = self.track_vertices_body(q_i, track_index)
        result: list[Any] = []
        for vertex in vertices:
            if is_symbolic(yaw):
                result.append(position + body_rotation @ vertex)
            else:
                result.append(np.asarray(position) + body_rotation @ vertex)
        return result

    def connector_vertices_world(self, state: Any) -> list[Any]:
        position = state[0:2]
        yaw = state[2]
        body_rotation = rotation_2d(yaw)
        result: list[Any] = []
        for vertex in self.connector_vertices_body():
            if is_symbolic(yaw):
                result.append(position + body_rotation @ as_column(vertex))
            else:
                result.append(np.asarray(position) + body_rotation @ vertex)
        return result

    def footprint_vertices_world(self, state: Any) -> list[Any]:
        vertices: list[Any] = []
        for track_index in range(2):
            vertices.extend(self.track_vertices_world(state, track_index))
        vertices.extend(self.connector_vertices_world(state))
        return vertices

    def support_vertices_world(self, state: Any) -> list[Any]:
        vertices: list[Any] = []
        for track_index in range(2):
            vertices.extend(self.track_vertices_world(state, track_index))
        return vertices

    def centre_of_mass_body(self, q: Any) -> Any:
        params = self.parameters
        symbolic = is_symbolic(q[0])
        if symbolic:
            weighted = params.body_mass * as_column(params.body_com)
        else:
            weighted = params.body_mass * params.body_com.copy()
        for index in range(2):
            weighted = weighted + params.track_mass * self.track_center_body(
                q[index], index
            )
        return weighted / (params.body_mass + 2.0 * params.track_mass)

    def centre_of_mass_world(self, state: Any) -> Any:
        position = state[0:2]
        yaw = state[2]
        q = state[3:5]
        body_com = self.centre_of_mass_body(q)
        if is_symbolic(yaw):
            return position + rotation_2d(yaw) @ body_com
        return np.asarray(position) + rotation_2d(yaw) @ body_com

    def support_projection_bounds(
        self,
        state: Any,
        normal_world: Any,
        epsilon: float,
    ) -> tuple[Any, Any]:
        projections = []
        for vertex in self.support_vertices_world(state):
            if is_symbolic(state[0]):
                projections.append(ca.dot(normal_world, vertex))
            else:
                projections.append(float(np.dot(normal_world, vertex)))
        lower = smooth_minimum(projections, epsilon)
        upper = smooth_maximum(projections, epsilon)
        return lower, upper

    def footprint_projection_bounds(
        self,
        state: Any,
        normal_world: Any,
        epsilon: float,
    ) -> tuple[Any, Any]:
        """Project the complete robot footprint on an arbitrary direction."""

        symbolic = is_symbolic(state[0]) or is_symbolic(normal_world[0])
        projections = []
        for vertex in self.footprint_vertices_world(state):
            if symbolic:
                projections.append(ca.dot(normal_world, vertex))
            else:
                projections.append(float(np.dot(normal_world, vertex)))
        if symbolic:
            return (
                smooth_minimum(projections, epsilon),
                smooth_maximum(projections, epsilon),
            )
        return min(projections), max(projections)

    def envelope_width_expression(
        self,
        state: Any,
        normal_world: Any,
        epsilon: float,
    ) -> Any:
        """Return the full formation width along ``normal_world``.

        The symbolic branch uses differentiable extrema so it can be used as
        the single clearance inequality in the NMPC.
        """

        lower, upper = self.footprint_projection_bounds(
            state,
            normal_world,
            epsilon,
        )
        return upper - lower

    def centred_envelope_width_expression(
        self,
        state: Any,
        centre_world: Any,
        normal_world: Any,
        epsilon: float,
    ) -> Any:
        """Width required around a prescribed corridor centreline.

        Unlike the minimum span, this quantity also accounts for lateral
        offset of an asymmetric formation: it is twice the furthest footprint
        projection from the planned centreline.
        """

        symbolic = is_symbolic(state[0]) or is_symbolic(normal_world[0])
        radii = []
        for vertex in self.footprint_vertices_world(state):
            if symbolic:
                offset = ca.dot(normal_world, vertex - centre_world)
                radii.append(smooth_abs(offset, epsilon))
            else:
                offset = float(np.dot(normal_world, vertex - centre_world))
                radii.append(abs(offset))
        radius = (
            smooth_maximum(radii, epsilon)
            if symbolic
            else max(radii)
        )
        return 2.0 * radius

    def lateral_stability_margins(
        self,
        state: Any,
        normal_world: Any,
        evaluation_point_world: Any | None,
        epsilon: float,
    ) -> tuple[Any, Any, Any]:
        lower, upper = self.support_projection_bounds(state, normal_world, epsilon)
        point = (
            self.centre_of_mass_world(state)
            if evaluation_point_world is None
            else evaluation_point_world
        )
        if is_symbolic(state[0]):
            projection = ca.dot(normal_world, point)
        else:
            projection = float(np.dot(normal_world, point))
        lower_margin = projection - lower
        upper_margin = upper - projection
        margin = 0.5 * (
            lower_margin
            + upper_margin
            - ((lower_margin - upper_margin) ** 2 + epsilon**2) ** 0.5
        )
        return lower_margin, upper_margin, margin

    def envelope_width(self, state: np.ndarray, normal_world: np.ndarray) -> float:
        return float(
            self.envelope_width_expression(
                state,
                normal_world,
                epsilon=0.0,
            )
        )
