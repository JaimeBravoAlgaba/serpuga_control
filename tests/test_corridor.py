import numpy as np

from serpuga_control.corridor import StraightGapCorridor


def test_corridor_width_profile_reaches_requested_values() -> None:
    corridor = StraightGapCorridor()
    outside = float(corridor.full_width(corridor.gap_start - 1.0))
    inside = float(corridor.full_width(0.5 * (corridor.gap_start + corridor.gap_end)))
    assert np.isclose(outside, corridor.open_width, atol=1.0e-5)
    assert np.isclose(inside, corridor.gap_width, atol=1.0e-5)


def test_corridor_preview_is_vectorised() -> None:
    corridor = StraightGapCorridor()
    values = corridor.preview(np.linspace(0.0, 3.0, 20))
    assert values.shape == (20,)
    assert np.all(values >= corridor.gap_width - 1.0e-9)


def test_clearance_residual_uses_physical_corridor_centre() -> None:
    corridor = StraightGapCorridor(
        open_width=1.0,
        gap_width=1.0,
        gap_start=10.0,
        gap_end=11.0,
        centre_y=0.25,
    )
    inside_vertices = [np.array([0.0, 0.25]), np.array([0.2, 0.65])]
    outside_vertices = [np.array([0.0, 0.25]), np.array([0.2, 0.80])]

    assert corridor.clearance_residual(inside_vertices, 0.02, 0.0) <= 0.0
    assert corridor.clearance_residual(outside_vertices, 0.02, 0.0) > 0.0


def test_clearance_residual_accounts_for_local_width_at_each_vertex() -> None:
    corridor = StraightGapCorridor(
        open_width=1.2,
        gap_width=0.5,
        gap_start=1.0,
        gap_end=2.0,
        transition_length=0.05,
    )
    vertices = [
        np.array([0.0, 0.24]),
        np.array([1.5, 0.24]),
    ]

    assert corridor.clearance_residual(vertices, 0.02, 0.0) > 0.0
