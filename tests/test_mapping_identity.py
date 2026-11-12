"""Ambiguous source/city joins must never depend on mapping row order."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.mapping_utils import find_city_in_mapping


@pytest.mark.parametrize("reverse", [False, True])
def test_same_name_cities_with_the_same_source_remain_ambiguous(reverse):
    rows = [
        {"city": "London", "country": "United Kingdom", "videos": "[shared_video]"},
        {"city": "London", "country": "Canada", "videos": "[shared_video]"},
    ]
    result = find_city_in_mapping("London", "shared_video", pd.DataFrame(rows[::-1] if reverse else rows))
    assert result is None


def test_unique_source_match_wins_over_filename_city():
    mapping = pd.DataFrame([
        {"city": "London", "country": "Canada", "videos": "[other_video]"},
        {"city": "New York", "country": "United States", "videos": "[shared_video]"},
    ])
    assert find_city_in_mapping("London", "shared_video", mapping)["city"] == "New York"


def test_duplicate_source_can_be_disambiguated_by_a_unique_city_name():
    mapping = pd.DataFrame([
        {"city": "London", "country": "United Kingdom", "videos": "[shared_video]"},
        {"city": "New York", "country": "United States", "videos": "[shared_video]"},
    ])
    assert find_city_in_mapping("NewYork", "shared_video", mapping)["city"] == "New York"


def test_ambiguous_source_cannot_fall_back_to_an_unrelated_name_match():
    mapping = pd.DataFrame([
        {"city": "London", "country": "United Kingdom", "videos": "[shared_video]"},
        {"city": "London", "country": "Canada", "videos": "[shared_video]"},
        {"city": "Paris", "country": "France", "videos": "[other_video]"},
    ])
    assert find_city_in_mapping("Paris", "shared_video", mapping) is None


def test_empty_source_cannot_match_an_empty_videos_list():
    mapping = pd.DataFrame([
        {"city": "Paris", "country": "France", "videos": "[]"},
        {"city": "London", "country": "United Kingdom", "videos": "[video]"},
    ])
    assert find_city_in_mapping("London", "", mapping)["city"] == "London"
