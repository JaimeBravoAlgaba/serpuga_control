from pathlib import Path

import numpy as np
import pytest

from serpuga_control import MPCParameters
from serpuga_control.configuration import (
    ConfigurationError,
    ConfigurationStore,
    configuration_from_form_values,
    configuration_to_form_values,
)

BUILTIN_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_builtin_yaml_profiles_are_complete_and_valid() -> None:
    store = ConfigurationStore(BUILTIN_CONFIGS)
    assert store.list_profiles() == ["default", "open-turn", "parallel-gap"]
    for name in store.list_profiles():
        configuration = store.load(name)
        configuration.validate()
        assert configuration.mpc.track_speed_limit > 0.0


def test_form_values_round_trip_and_allow_gap_editing() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    values = configuration_to_form_values(configuration)
    values["scenario.gap_width_m"] = "0.40"
    values["mpc.track_speed_limit_mps"] = "0.31"
    restored = configuration_from_form_values(values)
    assert np.isclose(restored.corridor.gap_width, 0.40)
    assert np.isclose(restored.mpc.track_speed_limit, 0.31)
    np.testing.assert_allclose(
        restored.simulation.initial_state,
        configuration.simulation.initial_state,
    )
    assert "mpc.parallelism_weight" in values
    assert "mpc.articulation_rate_limit_rps" in values
    assert "mpc.track_speed_limit_mps" in values
    assert "mpc.slip_weight" not in values
    assert "mpc.minimum_stability_margin_m" not in values


def test_legacy_gap_pose_fields_are_ignored() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    mapping = configuration.to_mapping()
    mapping["robot"]["q1_narrow_deg"] = -45.0
    mapping["robot"]["q2_narrow_deg"] = 135.0
    mapping["robot"]["narrow_body_yaw_deg"] = 30.0

    restored = configuration.from_mapping(mapping)
    saved = restored.to_mapping()["robot"]

    assert "q1_narrow_deg" not in saved
    assert "q2_narrow_deg" not in saved
    assert "narrow_body_yaw_deg" not in saved


def test_legacy_mpc_profile_maps_alignment_weight_to_parallelism() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    mapping = configuration.to_mapping()
    mapping["mpc"].pop("parallelism_weight")
    mapping["mpc"]["track_alignment_weight"] = 17.0

    restored = configuration.from_mapping(mapping)

    assert restored.mpc.parallelism_weight == 17.0


def test_legacy_profile_without_articulation_rate_limit_uses_default() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    mapping = configuration.to_mapping()
    mapping["mpc"].pop("articulation_rate_limit_rps")

    restored = configuration.from_mapping(mapping)

    assert restored.mpc.articulation_rate_limit == MPCParameters().articulation_rate_limit


def test_legacy_profile_without_track_speed_limit_uses_default() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    mapping = configuration.to_mapping()
    mapping["mpc"].pop("track_speed_limit_mps")

    restored = configuration.from_mapping(mapping)

    assert restored.mpc.track_speed_limit == MPCParameters().track_speed_limit


def test_nonpositive_track_speed_limit_is_rejected() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    values = configuration_to_form_values(configuration)
    values["mpc.track_speed_limit_mps"] = "0"
    with pytest.raises(ConfigurationError, match="track_speed_limit"):
        configuration_from_form_values(values)


def test_profile_can_be_saved_and_loaded(tmp_path) -> None:
    source = ConfigurationStore(BUILTIN_CONFIGS).load("parallel-gap")
    store = ConfigurationStore(tmp_path)
    path = store.save("Mi configuración", source)
    assert path.name == "mi-configuracion.yaml"
    loaded = store.load("mi-configuracion")
    np.testing.assert_allclose(
        loaded.robot.pivot_positions,
        source.robot.pivot_positions,
    )
    assert loaded.mpc.track_speed_limit == source.mpc.track_speed_limit


def test_invalid_initial_joint_angle_is_rejected() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    values = configuration_to_form_values(configuration)
    values["simulation.initial_q1_deg"] = "200"
    with pytest.raises(ConfigurationError, match="Initial q1/q2"):
        configuration_from_form_values(values)
