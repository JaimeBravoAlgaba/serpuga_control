from pathlib import Path

import numpy as np
import pytest

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


def test_form_values_round_trip_and_allow_gap_editing() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    values = configuration_to_form_values(configuration)
    values["scenario.gap_width_m"] = "0.40"
    restored = configuration_from_form_values(values)
    assert np.isclose(restored.corridor.gap_width, 0.40)
    np.testing.assert_allclose(
        restored.simulation.initial_state,
        configuration.simulation.initial_state,
    )


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


def test_invalid_initial_joint_angle_is_rejected() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    values = configuration_to_form_values(configuration)
    values["simulation.initial_q1_deg"] = "25"
    with pytest.raises(ConfigurationError, match="Initial q1/q2"):
        configuration_from_form_values(values)
