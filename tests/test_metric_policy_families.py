"""A component predictor cannot bypass its detector family's validity policy."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.quality import apply_metric_policy


@pytest.mark.parametrize("family,column", [
    ("clothing", "short_sleeved_shirt"), ("belongings", "backpack"), ("vehicle_type", "car"),
    ("accident", "police_car"), ("road_condition", "crack"),
])
@pytest.mark.parametrize("status,experimental,visible", [
    ("validated", False, True), ("experimental", False, False),
    ("experimental", True, True), ("invalid", True, False), ("invalid_legacy", True, False),
])
def test_family_policy_applies_to_component_flags(family, column, status, experimental, visible):
    frame = pd.DataFrame([{"eligible_for_comparison": True, column: 1,
                           "metric_validity": {family: {"status": status}}}])
    result = apply_metric_policy(frame, include_experimental=experimental)
    assert bool(result[column].notna().iloc[0]) is visible
    assert frame[column].iloc[0] == 1, "Raw observations must remain available for audit"


@pytest.mark.parametrize("family_status,component_status", [("invalid", "validated"), ("validated", "invalid")])
def test_component_and_family_must_both_be_allowed(family_status, component_status):
    frame = pd.DataFrame([{"eligible_for_comparison": True, "backpack": 1,
        "metric_validity": {"belongings": {"status": family_status}, "backpack": {"status": component_status}}}])
    assert apply_metric_policy(frame, include_experimental=True)["backpack"].isna().all()
