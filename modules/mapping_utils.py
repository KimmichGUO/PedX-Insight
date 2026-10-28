"""Joining analysed videos to mapping.csv city metadata (shared by both aggregators).

Two data-quality problems made the join silently lossy:

1. Name drift. PedX-Crawler names folders underscore-free and space-free (`NewYork1_...`),
   while mapping.csv says "New York". The exact-match join failed, so NewYork1/NewYork3
   were published with no country, continent or socio-economic covariates, as a separate
   city from "New York". `find_city_in_mapping` now falls back to a normalised match
   (case, spaces and punctuation ignored) and callers adopt the canonical mapping name.

2. Zero as missing. mapping.csv encodes unknown covariates as 0 (gmp is 0 in ~88% of
   rows, literacy_rate in ~36%), and rows appended by the crawler bridge leave country
   fields empty (Ulm has no iso3/continent). A zero literacy rate or median age is
   impossible, so `clean_mapping` turns those zeros into NA, then backfills country-level
   fields from the other rows of the same country -- only when they all agree on one
   value, so nothing city-specific is invented.
"""

import re

import pandas as pd

# Covariates for which 0 is not a possible value, only a missing-value code.
IMPOSSIBLE_ZERO_COLS = ["gmp", "population_city", "population_country", "traffic_mortality",
                        "literacy_rate", "avg_height", "med_age", "gini"]
# Fields that may be filled from same-country rows when those rows are unanimous.
COUNTRY_LEVEL_COLS = ["iso3", "continent", "population_country", "traffic_mortality",
                      "literacy_rate", "med_age", "avg_height", "gini"]


def normalise_city(name):
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def clean_mapping(mapping_df):
    df = mapping_df.copy()
    for col in IMPOSSIBLE_ZERO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").mask(lambda s: s == 0)
    if "country" in df.columns:
        for col in COUNTRY_LEVEL_COLS:
            if col not in df.columns:
                continue
            missing = df[col].isna() & df["country"].notna()
            if not missing.any():
                continue
            for country in df.loc[missing, "country"].unique():
                values = df.loc[(df["country"] == country) & df[col].notna(), col].unique()
                if len(values) == 1:
                    df.loc[missing & (df["country"] == country), col] = values[0]
    return df


def _videos_list(videos_str):
    if isinstance(videos_str, str):
        if videos_str.startswith("[") and videos_str.endswith("]"):
            videos_str = videos_str[1:-1]
        return [v.strip().strip("'\"") for v in videos_str.split(",")]
    return videos_str


def find_city_in_mapping(city, link, mapping_df):
    """Return the mapping.csv row for (city, link), or None.

    Exact name match first, then a normalised match. Among several rows with the same name
    (two Londons), the one whose `videos` list contains `link` wins.
    """
    # Prefer the source identity to a filename-derived spelling of a city.
    if "videos" in mapping_df.columns:
        linked = mapping_df[mapping_df["videos"].map(lambda raw: link in (_videos_list(raw) or []) if isinstance(raw, (str, list, tuple)) else False)]
        if len(linked) == 1:
            return linked.iloc[0]
    city_matches = mapping_df[mapping_df["city"] == city]
    if len(city_matches) == 0:
        norm = normalise_city(city)
        city_matches = mapping_df[mapping_df["city"].map(normalise_city) == norm]
        if len(city_matches):
            print(f"  Note: city '{city}' matched mapping.csv '{city_matches.iloc[0]['city']}' "
                  f"by normalised name")
    if len(city_matches) == 0:
        print(f"Warning: City {city} not found in mapping.csv")
        return None
    if len(city_matches) == 1:
        return city_matches.iloc[0]
    for _, row in city_matches.iterrows():
        try:
            if pd.isna(row["videos"]) if not isinstance(row["videos"], (list, tuple)) else False:
                continue
            if link in _videos_list(row["videos"]):
                return row
        except Exception as e:
            print(f"Error parsing 'videos' column: {e}, value: {row.get('videos')}")
    print(f"Warning: City {city} has multiple matches but no record contains link {link}")
    return None  # ambiguous countries must never be resolved by row order
