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

