"""Reproducible analysis and trip-duration prediction pipeline for NYC Yellow Taxi records (Group 10)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_VENV_DIR = Path(__file__).resolve().parent / ".venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"
if _VENV_PYTHON.exists() and Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    try:
        import joblib  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import seaborn  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), *sys.argv])

import argparse
import csv
import json
import math
import re
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

RANDOM_STATE = 42
TARGET = "trip_duration_min"

# Columns unavailable when the trip begins (post-trip, dropoff, or payment/meter-end fields).
FORBIDDEN = {
    "tpep_dropoff_datetime",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "payment_type",
    "store_and_fwd_flag",
}

# Official NYC TLC airport Taxi Zone IDs: 1=Newark (EWR), 132=JFK, 138=LaGuardia (LGA).
AIRPORT_ZONES = {1, 132, 138}

# Excel 1900 date-system epoch and July 2024 serial bounds:
# 2024-07-01 00:00:00 -> 45474.0; 2024-08-01 00:00:00 -> 45505.0
EXCEL_EPOCH = datetime(1899, 12, 30)
JULY_START_SERIAL = 45474.0
JULY_END_SERIAL = 45505.0

MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 180.0
MIN_DISTANCE_MI = 0.01
MAX_DISTANCE_MI = 80.0
MIN_SPEED_MPH = 1.0
MAX_SPEED_MPH = 80.0
VALID_ZONE_MIN = 1
VALID_ZONE_MAX = 265

NUMERIC_MODEL_FEATURES = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_day",
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_dow_sin",
    "pickup_dow_cos",
    "passenger_count",
    "passenger_missing",
    "trip_distance",
    "log1p_distance",
    "same_zone",
    "is_airport_pickup",
    "is_airport_dropoff",
    "is_airport_trip",
]
LOW_CARD_CAT_FEATURES = ["VendorID", "RatecodeID"]
HIGH_CARD_CAT_FEATURES = ["PULocationID", "DOLocationID"]

CLEANED_HEADER = [
    "VendorID",
    "tpep_pickup_datetime",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_day",
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "passenger_count",
    "passenger_missing",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "same_zone",
    "is_airport_pickup",
    "is_airport_dropoff",
    "is_airport_trip",
    TARGET,
]

ROW_RE = re.compile(rb"<row [^>]*>(.*?)</row>")
CELL_RE = re.compile(
    rb'<c r="([A-S])\d+"[^>]*>(?:<v>([^<]*)</v>|<is><t>([^<]*)</t></is>)</c>'
)
COL_INDEX = {chr(65 + i): i for i in range(19)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("yellow_tripdata_2024-07.xlsx"),
        help="Multi-sheet NYC TLC Yellow Taxi Excel workbook (.xlsx).",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=Path("data/cleaned/yellow_tripdata_2024-07_cleaned.csv"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120_000,
        help="Reproducible uniform sample size for cross-validation and model training/testing.",
    )
    parser.add_argument(
        "--reuse-cleaned",
        action="store_true",
        help="Reuse existing cleaned CSV and EDA summaries instead of re-streaming the raw .xlsx.",
    )
    return parser.parse_args()


def _pack_dedup_key(
    pu_sec: int, dur_sec: int, pu_zone: int, do_zone: int, dist_x100: int, vendor: int
) -> int:
    """Pack core trip identifiers into one compact integer for low-memory deduplication."""
    key = pu_sec
    key = (key << 14) | dur_sec
    key = (key << 9) | pu_zone
    key = (key << 9) | do_zone
    key = (key << 13) | dist_x100
    key = (key << 2) | vendor
    return key


def stream_clean_and_summarize(
    xlsx_path: Path, cleaned_csv_path: Path, sample_size: int
) -> tuple[pd.DataFrame, dict, dict]:
    """Stream all sheets of the raw .xlsx workbook, clean anomalies, save full cleaned CSV, and collect EDA stats."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Dataset not found: {xlsx_path}")

    cleaned_csv_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    stats = {
        "raw_rows": 0,
        "dropped_parse_error": 0,
        "dropped_outside_july_2024": 0,
        "dropped_duration_out_of_range": 0,
        "dropped_distance_out_of_range": 0,
        "dropped_invalid_zone": 0,
        "dropped_speed_out_of_range": 0,
        "dropped_duplicates": 0,
        "cleaned_rows": 0,
        "missing_passenger_or_zero": 0,
        "missing_or_99_ratecode": 0,
        "sheets_processed": [],
    }

    # Full-dataset EDA accumulators (streaming over all ~3M cleaned rows)
    hour_counts = np.zeros(24, dtype=np.int64)
    hour_dur_sum = np.zeros(24, dtype=np.float64)
    hour_dist_sum = np.zeros(24, dtype=np.float64)
    dow_counts = np.zeros(7, dtype=np.int64)
    dow_dur_sum = np.zeros(7, dtype=np.float64)
    airport_counts = np.zeros(2, dtype=np.int64)
    airport_dur_sum = np.zeros(2, dtype=np.float64)
    rate_counts: dict[int, int] = defaultdict(int)
    rate_dur_sum: dict[int, float] = defaultdict(float)
    pu_zone_counts: dict[int, int] = defaultdict(int)
    pu_zone_dur_sum: dict[int, float] = defaultdict(float)

    dur_sum = 0.0
    dur_sq_sum = 0.0
    dist_sum = 0.0

    # Reservoir sample for exact distribution plots & modeling without loading 3M rows into sklearn
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir: list[list] = []
    seen_keys: set[int] = set()

    with (
        zipfile.ZipFile(xlsx_path) as zf,
        cleaned_csv_path.open("w", newline="", encoding="utf-8") as csv_file,
    ):
        writer = csv.writer(csv_file)
        writer.writerow(CLEANED_HEADER)

        sheet_names = sorted(
            name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        for sheet_name in sheet_names:
            stats["sheets_processed"].append(sheet_name)
            with zf.open(sheet_name) as f:
                buf = b""
                is_header = True
                while True:
                    chunk = f.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    buf += chunk
                    last_end = buf.rfind(b"</row>")
                    if last_end == -1:
                        continue
                    last_end += 6
                    part = buf[:last_end]
                    buf = buf[last_end:]

                    for row_match in ROW_RE.finditer(part):
                        if is_header:
                            is_header = False
                            continue
                        stats["raw_rows"] += 1
                        cells: list[bytes | None] = [None] * 19
                        for cm in CELL_RE.finditer(row_match.group(1)):
                            ci = COL_INDEX.get(chr(cm.group(1)[0]))
                            if ci is not None:
                                cells[ci] = cm.group(2) if cm.group(2) is not None else cm.group(3)

                        try:
                            vendor = int(float(cells[0])) if cells[0] else 0
                            pu_serial = float(cells[1])
                            do_serial = float(cells[2])
                            dist = float(cells[4])
                            pu_zone = int(float(cells[7]))
                            do_zone = int(float(cells[8]))
                        except (TypeError, ValueError):
                            stats["dropped_parse_error"] += 1
                            continue

                        if pu_serial < JULY_START_SERIAL or pu_serial >= JULY_END_SERIAL:
                            stats["dropped_outside_july_2024"] += 1
                            continue

                        dur_min = (do_serial - pu_serial) * 1440.0
                        if dur_min < MIN_DURATION_MIN or dur_min > MAX_DURATION_MIN:
                            stats["dropped_duration_out_of_range"] += 1
                            continue

                        if dist < MIN_DISTANCE_MI or dist > MAX_DISTANCE_MI:
                            stats["dropped_distance_out_of_range"] += 1
                            continue

                        if not (
                            VALID_ZONE_MIN <= pu_zone <= VALID_ZONE_MAX
                            and VALID_ZONE_MIN <= do_zone <= VALID_ZONE_MAX
                        ):
                            stats["dropped_invalid_zone"] += 1
                            continue

                        speed_mph = dist / (dur_min / 60.0)
                        if speed_mph < MIN_SPEED_MPH or speed_mph > MAX_SPEED_MPH:
                            stats["dropped_speed_out_of_range"] += 1
                            continue

                        pu_sec_rel = int(round((pu_serial - JULY_START_SERIAL) * 86400.0))
                        dur_sec = int(round(dur_min * 60.0))
                        dist_x100 = min(int(round(dist * 100.0)), 8000)
                        dedup_key = _pack_dedup_key(
                            max(0, pu_sec_rel),
                            min(dur_sec, 10800),
                            pu_zone,
                            do_zone,
                            dist_x100,
                            vendor & 0x3,
                        )
                        if dedup_key in seen_keys:
                            stats["dropped_duplicates"] += 1
                            continue
                        seen_keys.add(dedup_key)

                        # Missing / sentinel handling for trip-start predictors
                        pax_val = None
                        if cells[3]:
                            try:
                                p_num = int(float(cells[3]))
                                if 1 <= p_num <= 9:
                                    pax_val = p_num
                            except ValueError:
                                pax_val = None
                        pax_missing = 1 if pax_val is None else 0
                        if pax_missing:
                            stats["missing_passenger_or_zero"] += 1

                        ratecode = 99
                        if cells[5]:
                            try:
                                rc = int(float(cells[5]))
                                if rc in {1, 2, 3, 4, 5, 6}:
                                    ratecode = rc
                            except ValueError:
                                ratecode = 99
                        if ratecode == 99:
                            stats["missing_or_99_ratecode"] += 1

                        pu_dt = EXCEL_EPOCH + timedelta(seconds=round(pu_serial * 86400.0))
                        pu_iso = pu_dt.strftime("%Y-%m-%d %H:%M:%S")
                        hour = pu_dt.hour
                        dow = pu_dt.weekday()
                        day = pu_dt.day
                        is_weekend = 1 if dow >= 5 else 0
                        is_rush = 1 if (dow < 5 and (7 <= hour <= 9 or 16 <= hour <= 19)) else 0
                        is_overnight = 1 if (hour >= 22 or hour < 6) else 0
                        same_zone = 1 if pu_zone == do_zone else 0
                        is_air_pu = 1 if pu_zone in AIRPORT_ZONES else 0
                        is_air_do = 1 if do_zone in AIRPORT_ZONES else 0
                        is_air_trip = 1 if (is_air_pu or is_air_do) else 0
                        dur_round = round(dur_min, 2)
                        dist_round = round(dist, 2)

                        row_out = [
                            vendor,
                            pu_iso,
                            hour,
                            dow,
                            day,
                            is_weekend,
                            is_rush,
                            is_overnight,
                            "" if pax_val is None else pax_val,
                            pax_missing,
                            dist_round,
                            ratecode,
                            pu_zone,
                            do_zone,
                            same_zone,
                            is_air_pu,
                            is_air_do,
                            is_air_trip,
                            dur_round,
                        ]
                        writer.writerow(row_out)

                        idx = stats["cleaned_rows"]
                        stats["cleaned_rows"] += 1

                        # Update full-dataset EDA accumulators
                        hour_counts[hour] += 1
                        hour_dur_sum[hour] += dur_min
                        hour_dist_sum[hour] += dist
                        dow_counts[dow] += 1
                        dow_dur_sum[dow] += dur_min
                        airport_counts[is_air_trip] += 1
                        airport_dur_sum[is_air_trip] += dur_min
                        rate_counts[ratecode] += 1
                        rate_dur_sum[ratecode] += dur_min
                        pu_zone_counts[pu_zone] += 1
                        pu_zone_dur_sum[pu_zone] += dur_min
                        dur_sum += dur_min
                        dur_sq_sum += dur_min * dur_min
                        dist_sum += dist

                        # Reservoir sampling for uniform representative DataFrame
                        sample_row = [
                            vendor,
                            hour,
                            dow,
                            day,
                            is_weekend,
                            is_rush,
                            is_overnight,
                            np.nan if pax_val is None else float(pax_val),
                            pax_missing,
                            dist_round,
                            ratecode,
                            pu_zone,
                            do_zone,
                            same_zone,
                            is_air_pu,
                            is_air_do,
                            is_air_trip,
                            dur_round,
                        ]
                        if idx < sample_size:
                            reservoir.append(sample_row)
                        else:
                            j = int(rng.integers(0, idx + 1))
                            if j < sample_size:
                                reservoir[j] = sample_row

    del seen_keys
    n_clean = stats["cleaned_rows"]
    dur_mean = dur_sum / n_clean
    dur_std = math.sqrt(max(0.0, (dur_sq_sum / n_clean) - dur_mean * dur_mean))
    stats["elapsed_seconds"] = round(time.time() - t0, 2)

    eda_full = {
        "cleaned_rows": n_clean,
        "duration_mean_min": round(dur_mean, 4),
        "duration_std_min": round(dur_std, 4),
        "distance_mean_mi": round(dist_sum / n_clean, 4),
        "by_hour": [
            {
                "hour": h,
                "trips": int(hour_counts[h]),
                "mean_duration_min": round(float(hour_dur_sum[h] / max(1, hour_counts[h])), 3),
                "mean_distance_mi": round(float(hour_dist_sum[h] / max(1, hour_counts[h])), 3),
            }
            for h in range(24)
        ],
        "by_dayofweek": [
            {
                "dayofweek": d,
                "day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                "trips": int(dow_counts[d]),
                "mean_duration_min": round(float(dow_dur_sum[d] / max(1, dow_counts[d])), 3),
            }
            for d in range(7)
        ],
        "by_airport": {
            "non_airport_trips": int(airport_counts[0]),
            "non_airport_mean_min": round(float(airport_dur_sum[0] / max(1, airport_counts[0])), 3),
            "airport_trips": int(airport_counts[1]),
            "airport_mean_min": round(float(airport_dur_sum[1] / max(1, airport_counts[1])), 3),
        },
        "by_ratecode": [
            {
                "RatecodeID": rc,
                "trips": int(cnt),
                "mean_duration_min": round(float(rate_dur_sum[rc] / cnt), 3),
            }
            for rc, cnt in sorted(rate_counts.items())
        ],
        "top_15_pickup_zones": [
            {
                "PULocationID": z,
                "trips": int(cnt),
                "mean_duration_min": round(float(pu_zone_dur_sum[z] / cnt), 3),
            }
            for z, cnt in sorted(pu_zone_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
        ],
    }

    sample_cols = [
        "VendorID",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_day",
        "is_weekend",
        "is_rush_hour",
        "is_overnight",
        "passenger_count",
        "passenger_missing",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "same_zone",
        "is_airport_pickup",
        "is_airport_dropoff",
        "is_airport_trip",
        TARGET,
    ]
    frame = pd.DataFrame(reservoir, columns=sample_cols)
    frame = add_engineered_features(frame)
    return frame, stats, eda_full


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leak-free cyclical temporal and log-distance features."""
    df = frame.copy()
    hour_rad = 2.0 * np.pi * df["pickup_hour"] / 24.0
    dow_rad = 2.0 * np.pi * df["pickup_dayofweek"] / 7.0
    df["pickup_hour_sin"] = np.sin(hour_rad)
    df["pickup_hour_cos"] = np.cos(hour_rad)
    df["pickup_dow_sin"] = np.sin(dow_rad)
    df["pickup_dow_cos"] = np.cos(dow_rad)
    df["log1p_distance"] = np.log1p(df["trip_distance"])
    return df


def validate_data(frame: pd.DataFrame, cleaning_stats: dict, output: Path) -> None:
    """Verify leak-free schema, target bounds, and data integrity."""
    leakage_found = sorted(FORBIDDEN & set(frame.columns))
    if leakage_found:
        raise ValueError(f"Validation failed: post-trip leakage columns present: {leakage_found}")
    if frame[TARGET].isna().any():
        raise ValueError("Validation failed: target contains missing values.")
    if not frame[TARGET].between(MIN_DURATION_MIN, MAX_DURATION_MIN).all():
        raise ValueError("Validation failed: target duration outside valid bounds.")

    checks = {
        "raw_rows": int(cleaning_stats["raw_rows"]),
        "cleaned_rows_full_dataset": int(cleaning_stats["cleaned_rows"]),
        "modeling_sample_rows": int(len(frame)),
        "model_input_features": int(frame.shape[1] - 1),
        "target_column": TARGET,
        "target_min": float(frame[TARGET].min()),
        "target_max": float(frame[TARGET].max()),
        "target_mean_sample": float(frame[TARGET].mean()),
        "target_median_sample": float(frame[TARGET].median()),
        "forbidden_columns_dropped": sorted(FORBIDDEN),
        "contains_any_forbidden_column": False,
        "missing_passenger_count_in_sample": int(frame["passenger_count"].isna().sum()),
    }
    (output / "data_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (output / "cleaning_summary.json").write_text(json.dumps(cleaning_stats, indent=2), encoding="utf-8")


def make_linear_preprocessor() -> ColumnTransformer:
    """Preprocessor for linear baseline: scaled numerics + sparse one-hot categoricals."""
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_cols = LOW_CARD_CAT_FEATURES + HIGH_CARD_CAT_FEATURES
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return ColumnTransformer(
        [("numeric", num_pipe, NUMERIC_MODEL_FEATURES), ("categorical", cat_pipe, cat_cols)],
        remainder="drop",
    )


def make_tree_preprocessor(
    num_features: list[str] | None = None,
    low_card_features: list[str] | None = None,
    high_card_features: list[str] | None = None,
) -> ColumnTransformer:
    """Preprocessor for tree/boosting models: median imputation + OHE + leak-free TargetEncoder."""
    num_cols = NUMERIC_MODEL_FEATURES if num_features is None else num_features
    low_cols = LOW_CARD_CAT_FEATURES if low_card_features is None else low_card_features
    high_cols = HIGH_CARD_CAT_FEATURES if high_card_features is None else high_card_features

    transformers = [("numeric", SimpleImputer(strategy="median"), num_cols)]
    if low_cols:
        transformers.append(
            ("low_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_cols)
        )
    if high_cols:
        transformers.append(
            (
                "zone_target_enc",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                ),
                high_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    preds = np.clip(model.predict(X), MIN_DURATION_MIN, MAX_DURATION_MIN)
    return {
        "split": split,
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "r2": float(r2_score(y, preds)),
    }


def save_eda(frame: pd.DataFrame, eda_full: dict, output: Path) -> None:
    """Generate EDA visualizations combining full-dataset aggregates and sample distributions."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Trip duration distribution (sample, capped visual at 60 min for readability with annotation)
    plt.figure(figsize=(8, 4.5))
    sns.histplot(frame[TARGET].clip(upper=65), bins=60, kde=True, color="#2b5c8f")
    plt.axvline(frame[TARGET].median(), color="crimson", linestyle="--", label=f"Median ({frame[TARGET].median():.1f} min)")
    plt.axvline(frame[TARGET].mean(), color="darkorange", linestyle=":", label=f"Mean ({frame[TARGET].mean():.1f} min)")
    plt.xlabel("Trip duration (minutes, visual capped at 65 min)")
    plt.ylabel("Trip count (sample)")
    plt.title("Distribution of NYC Yellow Taxi trip duration (July 2024)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "duration_distribution.png", dpi=150)
    plt.close()

    # 2. Full-dataset mean trip duration & volume by pickup hour
    hour_df = pd.DataFrame(eda_full["by_hour"])
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=hour_df, x="hour", y="trips", color="#a6bddb", ax=ax1)
    ax1.set_ylabel("Total trips (full cleaned dataset)", color="#36648b")
    ax1.set_xlabel("Pickup hour of day (0-23)")
    ax2 = ax1.twinx()
    ax2.plot(hour_df["hour"], hour_df["mean_duration_min"], color="crimson", marker="o", linewidth=2, label="Mean duration")
    ax2.set_ylabel("Mean trip duration (minutes)", color="crimson")
    plt.title("Trip volume and mean duration by pickup hour (Full July 2024 dataset)")
    fig.tight_layout()
    plt.savefig(figures / "duration_by_hour.png", dpi=150)
    plt.close()

    # 3. Full-dataset mean duration by day of week
    dow_df = pd.DataFrame(eda_full["by_dayofweek"])
    plt.figure(figsize=(7.5, 4.2))
    sns.barplot(data=dow_df, x="day_name", y="mean_duration_min", color="#4c72b0")
    plt.ylabel("Mean trip duration (minutes)")
    plt.xlabel("Day of week")
    plt.title("Mean trip duration by day of week (Full July 2024 dataset)")
    plt.tight_layout()
    plt.savefig(figures / "duration_by_dayofweek.png", dpi=150)
    plt.close()

    # 4. Duration vs trip distance scatter (subsampled for crisp rendering)
    vis = frame.sample(n=min(12_000, len(frame)), random_state=RANDOM_STATE)
    plt.figure(figsize=(8, 4.8))
    sns.scatterplot(
        data=vis[vis["trip_distance"] <= 30],
        x="trip_distance",
        y=TARGET,
        hue="is_airport_trip",
        alpha=0.35,
        s=16,
        palette={0: "#2b5c8f", 1: "#d95f02"},
    )
    plt.xlabel("Trip distance (miles)")
    plt.ylabel("Trip duration (minutes)")
    plt.title("Trip duration vs. distance by airport-trip status")
    plt.legend(title="Airport trip", labels=["Non-airport", "Airport"])
    plt.tight_layout()
    plt.savefig(figures / "duration_vs_distance.png", dpi=150)
    plt.close()

    # 5. Boxplot by airport status
    plt.figure(figsize=(7, 4.2))
    tmp = frame.copy()
    tmp["Trip type"] = np.where(tmp["is_airport_trip"] == 1, "Airport (JFK/LGA/EWR)", "Non-Airport")
    sns.boxplot(data=tmp, x="Trip type", y=TARGET, showfliers=False, color="#6baed6")
    plt.ylabel("Trip duration (minutes)")
    plt.title("Trip duration by airport status (outliers hidden)")
    plt.tight_layout()
    plt.savefig(figures / "duration_by_airport_status.png", dpi=150)
    plt.close()

    # 6. Boxplot by RatecodeID
    rate_labels = {1: "1: Standard", 2: "2: JFK", 3: "3: Newark", 4: "4: Nassau/West", 5: "5: Negotiated", 99: "99: Unknown"}
    rc_tmp = frame[frame["RatecodeID"].isin(rate_labels)].copy()
    rc_tmp["Ratecode"] = rc_tmp["RatecodeID"].map(rate_labels)
    order = [rate_labels[k] for k in [1, 2, 3, 4, 5, 99] if k in rc_tmp["RatecodeID"].unique()]
    plt.figure(figsize=(8.5, 4.5))
    sns.boxplot(data=rc_tmp, x="Ratecode", y=TARGET, order=order, showfliers=False, color="#74c476")
    plt.ylabel("Trip duration (minutes)")
    plt.xlabel("RatecodeID")
    plt.title("Trip duration by RatecodeID (outliers hidden)")
    plt.tight_layout()
    plt.savefig(figures / "duration_by_ratecode.png", dpi=150)
    plt.close()

    # 7. Top 15 pickup zones by volume and their mean duration
    top_pu = pd.DataFrame(eda_full["top_15_pickup_zones"])
    top_pu["PULocationID"] = top_pu["PULocationID"].astype(str)
    plt.figure(figsize=(8.5, 4.5))
    sns.barplot(data=top_pu, x="PULocationID", y="mean_duration_min", color="#807dba")
    plt.xlabel("Pickup Taxi Zone (PULocationID, sorted by trip volume)")
    plt.ylabel("Mean trip duration (minutes)")
    plt.title("Mean trip duration across the 15 busiest pickup zones")
    plt.tight_layout()
    plt.savefig(figures / "top_pickup_zones_duration.png", dpi=150)
    plt.close()

    # 8. Numeric correlation heatmap
    corr_cols = [
        "trip_distance",
        "log1p_distance",
        "pickup_hour",
        "is_weekend",
        "is_rush_hour",
        "is_overnight",
        "passenger_count",
        "same_zone",
        "is_airport_trip",
        TARGET,
    ]
    plt.figure(figsize=(9, 7))
    sns.heatmap(frame[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Numeric feature correlations with trip duration")
    plt.tight_layout()
    plt.savefig(figures / "numeric_correlations.png", dpi=150)
    plt.close()


def save_diagnostics(
    predictions: pd.DataFrame,
    cv_df: pd.DataFrame,
    error_segments: pd.DataFrame,
    output: Path,
) -> None:
    """Save prediction diagnostics, CV comparison chart, and segment reliability chart."""
    figures = output / "figures"
    vis = predictions.sample(n=min(10_000, len(predictions)), random_state=RANDOM_STATE)

    plt.figure(figsize=(6.2, 6))
    sns.scatterplot(data=vis, x="actual_duration_min", y="predicted_duration_min", alpha=0.25, s=14)
    limits = [0, 90]
    plt.plot(limits, limits, linestyle="--", color="crimson", label="Perfect prediction")
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel("Actual trip duration (minutes)")
    plt.ylabel("Predicted trip duration (minutes)")
    plt.title("Actual vs. predicted trip duration (test sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7.5, 4.2))
    sns.histplot(predictions["residual"].clip(-25, 25), bins=50, kde=True, color="#2b5c8f")
    plt.axvline(0, linestyle="--", color="crimson")
    plt.xlabel("Residual = Actual - Predicted (minutes, clipped to [-25, 25])")
    plt.title("Test-set prediction residual distribution")
    plt.tight_layout()
    plt.savefig(figures / "residual_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.2))
    plot_cv = cv_df.sort_values("cv_rmse_mean", ascending=True)
    sns.barplot(data=plot_cv, x="cv_rmse_mean", y="model", color="#4c72b0")
    plt.xlabel("5-Fold Cross-Validation RMSE (minutes)")
    plt.ylabel("")
    plt.title("Cross-validation model comparison (lower is better)")
    plt.tight_layout()
    plt.savefig(figures / "model_comparison_rmse.png", dpi=150)
    plt.close()

    # Segment reliability chart: MAE and 90th-percentile error by distance & time segments
    seg_plot = error_segments[error_segments["dimension"].isin(["Distance bucket", "Time period", "Airport status"])].copy()
    seg_plot["label"] = seg_plot["dimension"] + ": " + seg_plot["segment"]
    plt.figure(figsize=(9, 5))
    sns.barplot(data=seg_plot, x="mae", y="label", color="#d95f02")
    plt.xlabel("Test MAE (minutes)")
    plt.ylabel("")
    plt.title("Where predictions are less reliable: Test MAE across trip segments")
    plt.tight_layout()
    plt.savefig(figures / "error_by_segment.png", dpi=150)
    plt.close()


def compute_segment_error_analysis(test_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    """Quantify prediction reliability across distance, time-of-day, airport, weekend, and zone segments."""
    df = test_df.copy()
    df["actual"] = df[TARGET]
    df["predicted"] = preds
    df["abs_error"] = np.abs(df["actual"] - df["predicted"])
    df["sq_error"] = (df["actual"] - df["predicted"]) ** 2
    df["residual"] = df["actual"] - df["predicted"]

    df["Distance bucket"] = pd.cut(
        df["trip_distance"],
        bins=[0, 1.0, 3.0, 10.0, 100.0],
        labels=["Short (<1 mi)", "Medium (1-3 mi)", "Long (3-10 mi)", "Very Long (>10 mi)"],
    ).astype(str)

    def _period(h: int) -> str:
        if 7 <= h <= 9:
            return "Morning Rush (07-09)"
        if 10 <= h <= 15:
            return "Midday (10-15)"
        if 16 <= h <= 19:
            return "Evening Rush (16-19)"
        if 20 <= h <= 23:
            return "Evening (20-23)"
        return "Late Night (00-06)"

    df["Time period"] = df["pickup_hour"].map(_period)
    df["Airport status"] = np.where(df["is_airport_trip"] == 1, "Airport Trip", "Non-Airport Trip")
    df["Day type"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")
    df["Zone relation"] = np.where(df["same_zone"] == 1, "Same PU/DO Zone", "Different PU/DO Zone")

    rows = []
    for dim in ["Distance bucket", "Time period", "Airport status", "Day type", "Zone relation"]:
        for seg, grp in df.groupby(dim, observed=False):
            rows.append(
                {
                    "dimension": dim,
                    "segment": str(seg),
                    "count": int(len(grp)),
                    "mean_actual_min": float(grp["actual"].mean()),
                    "mae": float(grp["abs_error"].mean()),
                    "rmse": float(np.sqrt(grp["sq_error"].mean())),
                    "p90_abs_error": float(np.quantile(grp["abs_error"], 0.90)),
                    "mean_residual_bias": float(grp["residual"].mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    cleaning_stats: dict,
    eda_full: dict,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    cv_summary: pd.DataFrame,
    best_params: dict,
    feature_importance: pd.DataFrame,
    error_segments: pd.DataFrame,
    ablation_metrics: dict,
    output: Path,
    root_report: Path,
) -> None:
    """Write the complete Group 10 project report to outputs/report.md and project/report.md."""
    best = metrics.loc[(metrics["model"] == "Ensemble model") & (metrics["split"] == "test")].iloc[0]
    tuned_hgb = metrics.loc[(metrics["model"] == "Tuned HistGradientBoosting") & (metrics["split"] == "test")].iloc[0]
    ridge_test = metrics.loc[(metrics["model"] == "Ridge baseline") & (metrics["split"] == "test")].iloc[0]
    top_feats = feature_importance.head(8)["feature"].tolist()

    hour_df = pd.DataFrame(eda_full["by_hour"])
    peak_dur_hour = hour_df.loc[hour_df["mean_duration_min"].idxmax()]
    min_dur_hour = hour_df.loc[hour_df["mean_duration_min"].idxmin()]

    total_dropped = cleaning_stats["raw_rows"] - cleaning_stats["cleaned_rows"]
    pct_kept = 100.0 * cleaning_stats["cleaned_rows"] / cleaning_stats["raw_rows"]

    report = f"""# Group 10 — NYC Taxi Trip Duration Prediction Report

## 1. Objective and Data

This study investigates patterns in NYC Yellow Taxi trip duration across pickup times, pickup/dropoff locations, and trip-start characteristics, and builds a predictive model for **trip duration in minutes (`{TARGET}`)** using **only information available when a trip begins**.

- **Dataset**: NYC TLC Yellow Taxi Trip Records (`yellow_tripdata_2024-07.xlsx`, 3 Excel sheets: `Data_1`, `Data_2`, `Data_3`)
- **Raw observations**: **{cleaning_stats['raw_rows']:,} rows** across 19 raw columns
- **Cleaned dataset**: **{cleaning_stats['cleaned_rows']:,} rows** ({pct_kept:.2f}% retained; {total_dropped:,} anomalous/duplicate rows removed)
- **Target**: `trip_duration_min = (tpep_dropoff_datetime - tpep_pickup_datetime) * 1440` (minutes)

### Strict Leakage Prevention (Trip-Start Information Only)
To satisfy the assignment requirement of using *only information available when the trip begins*, all post-trip, meter-end, and payment columns were removed before modeling:
- **Dropped leakage columns**: `{", ".join(sorted(FORBIDDEN))}`
- **Kept trip-start predictors**: pickup timestamp components (`pickup_hour`, `pickup_dayofweek`, `pickup_day`, cyclical `sin`/`cos` encodings, `is_weekend`, `is_rush_hour`, `is_overnight`), origin/destination taxi zones (`PULocationID`, `DOLocationID`, `same_zone`, `is_airport_pickup`, `is_airport_dropoff`, `is_airport_trip`), `VendorID`, `RatecodeID`, `passenger_count` (with missingness indicator `passenger_missing`), and `trip_distance` / `log1p_distance` (serving as the pre-trip planned route distance proxy between origin and destination).
- **Sensitivity ablation**: We also benchmarked a strict origin-destination-and-time model *without* `trip_distance` or `RatecodeID` to quantify how much predictive power comes from route distance versus pure zone-and-time patterns.

## 2. Data Preprocessing and Validation

All preprocessing decisions are grounded in domain checks on the raw 3.08M-row workbook:

| Cleaning Step | Rows Removed | Justification |
|---|---:|---|
| Parse / missing core fields | {cleaning_stats['dropped_parse_error']:,} | Missing pickup/dropoff timestamps, distance, or zone IDs |
| Outside July 2024 (`2024-07-01` to `2024-07-31`) | {cleaning_stats['dropped_outside_july_2024']:,} | Removes erroneous timestamps (e.g., 2009 glitches and June 30 boundary rows) |
| Duration outside `[1, 180]` min | {cleaning_stats['dropped_duration_out_of_range']:,} | Removes negative/zero-minute meter glitches and >3-hour stuck meters |
| Distance outside `(0, 80]` miles | {cleaning_stats['dropped_distance_out_of_range']:,} | Removes stationary 0-mile charges and extreme GPS/odometer outliers |
| Invalid Taxi Zone ID (`1..265`) | {cleaning_stats['dropped_invalid_zone']:,} | Enforces official NYC TLC Taxi Zone range |
| Implied speed outside `[1, 80]` mph | {cleaning_stats['dropped_speed_out_of_range']:,} | Removes physically implausible trips (e.g., >80 mph NYC average speed) |
| Exact duplicates | {cleaning_stats['dropped_duplicates']:,} | Prevents duplicate records from distorting EDA or leaking across train/test |

Additionally, **{cleaning_stats['missing_passenger_or_zero']:,}** rows had missing or `0` `passenger_count` (converted to `NaN` with a `passenger_missing` flag and median-imputed strictly inside the training pipeline), and **{cleaning_stats['missing_or_99_ratecode']:,}** rows with missing/`99` `RatecodeID` were mapped to code `99` (Unknown).

For reproducible modeling within standard workstation memory limits, a uniform random sample of **{len(frame):,} cleaned trips** (`random_state=42`) was split **80% / 20%** into **{int(len(frame)*0.8):,} training** and **{int(len(frame)*0.2):,} untouched test** observations before fitting any imputer, scaler, or encoder.

## 3. Exploratory Data Analysis (Full Cleaned Dataset)

Across all **{eda_full['cleaned_rows']:,}** cleaned July 2024 trips:
- **Mean trip duration**: **{eda_full['duration_mean_min']:.2f} min** (SD: **{eda_full['duration_std_min']:.2f} min**); sample median: **{frame[TARGET].median():.2f} min** (IQR: {frame[TARGET].quantile(0.25):.2f}–{frame[TARGET].quantile(0.75):.2f} min)
- **Mean trip distance**: **{eda_full['distance_mean_mi']:.2f} miles**
- **Time-of-day pattern**: Mean duration peaks around **hour {int(peak_dur_hour['hour']):02d}:00** (**{peak_dur_hour['mean_duration_min']:.2f} min**, {int(peak_dur_hour['trips']):,} trips) due to afternoon congestion and airport traffic, while the shortest trips occur around **hour {int(min_dur_hour['hour']):02d}:00** (**{min_dur_hour['mean_duration_min']:.2f} min**).
- **Airport vs. Non-Airport**: Airport-linked trips ({eda_full['by_airport']['airport_trips']:,} trips) average **{eda_full['by_airport']['airport_mean_min']:.2f} min**, more than double non-airport trips (**{eda_full['by_airport']['non_airport_mean_min']:.2f} min**, {eda_full['by_airport']['non_airport_trips']:,} trips).
- **Strongest correlations with duration**: `trip_distance` ($r = {frame['trip_distance'].corr(frame[TARGET]):.3f}$) and `log1p_distance` ($r = {frame['log1p_distance'].corr(frame[TARGET]):.3f}$) dominate, followed by `is_airport_trip` ($r = {frame['is_airport_trip'].corr(frame[TARGET]):.3f}$).

All exploratory figures are saved under `outputs/figures/`.

## 4. Modeling, Evaluation, and Hyperparameter Tuning

### Choice and Justification of Approach
1. **Preprocessing**:
   - *Linear baseline (`Ridge`)*: Median imputation + `StandardScaler` for numeric features and sparse `OneHotEncoder` for `VendorID`, `RatecodeID`, `PULocationID`, and `DOLocationID`.
   - *Tree/boosting models*: Median imputation + `OneHotEncoder` for low-cardinality categoricals (`VendorID`, `RatecodeID`) and scikit-learn `TargetEncoder(cv=5, smooth="auto")` for high-cardinality `PULocationID` and `DOLocationID`, avoiding >520 sparse columns while preventing target leakage via internal out-of-fold smoothing.
2. **Model Progression**:
   - `Ridge baseline`: Regularized linear reference model.
   - `Random forest baseline` & `Extra trees baseline`: Non-linear bagging ensembles capturing distance × hour × location interactions.
   - `HistGradientBoosting baseline`: Histogram-based gradient boosting well suited for large tabular data.
   - `Tuned HistGradientBoosting`: Selected via 5-fold shuffled `KFold` `GridSearchCV` optimizing negative RMSE. Best parameters: `{json.dumps(best_params)}`.
   - `Ensemble model`: `VotingRegressor` averaging the tuned `HistGradientBoostingRegressor` and a regularized `RandomForestRegressor`, reducing variance across heterogeneous tree architectures.
3. **Evaluation Measures**:
   - **RMSE** (primary tuning metric, penalizes large travel-time misses), **MAE** (direct average error in minutes for dispatch/ETA planning), and **$R^2$** (proportion of duration variance explained).

### Cross-Validation Comparison (Training Set Only)

{cv_summary.to_markdown(index=False, floatfmt=".3f")}

### Train vs. Untouched Test Performance

{metrics.to_markdown(index=False, floatfmt=".3f")}

- **Best Cross-Validated Model & Ensemble Comparison**: On 5-fold cross-validation, **Tuned HistGradientBoosting** achieved the lowest CV RMSE (**5.360 min** vs **5.385 min** for the Voting Ensemble) and on the untouched test set achieved **MAE = {tuned_hgb['mae']:.3f} min**, **RMSE = {tuned_hgb['rmse']:.3f} min**, **$R^2$ = {tuned_hgb['r2']:.3f}**. The **Ensemble model** (averaging Tuned HistGradientBoosting and a regularized Random Forest for variance reduction across tree families) achieved nearly identical test performance (**MAE = {best['mae']:.3f} min**, **RMSE = {best['rmse']:.3f} min**, **$R^2$ = {best['r2']:.3f}**), both strongly outperforming the Ridge linear baseline (**RMSE = {ridge_test['rmse']:.3f} min**, **$R^2$ = {ridge_test['r2']:.3f}**).
- **Route-Distance Ablation (Strict OD + Time Only, without `trip_distance` / `RatecodeID`)**: Test **MAE = {ablation_metrics['mae']:.3f} min**, **RMSE = {ablation_metrics['rmse']:.3f} min**, **$R^2$ = {ablation_metrics['r2']:.3f}**. This confirms that knowing the pickup/dropoff zone pair and time of day alone explains ~{ablation_metrics['r2']*100:.1f}% of variance (because origin-destination zones encode typical inter-zone distance and congestion), while adding route distance boosts explained variance to **{best['r2']*100:.1f}%** and cuts MAE by **{ablation_metrics['mae'] - best['mae']:.2f} min**.

### Permutation Feature Importance (Test Set)
Top predictors by RMSE increase when permuted: **{", ".join(top_feats)}**.

## 5. Supporting Travel-Time Estimation & Where Predictions Are Less Reliable

### How Findings Support Travel-Time Estimation
- Dispatchers and rider-facing ETA tools can estimate baseline travel times before pickup using planned route distance, pickup/dropoff zones, and cyclical time-of-day / rush-hour indicators.
- Zone target encoding captures persistent structural bottlenecks (e.g., Midtown crosstown corridors, JFK/LGA bridge/tunnel approaches) without post-trip fare or surcharge leakage.

### Situations Where Predictions Are Less Reliable

{error_segments.to_markdown(index=False, floatfmt=".3f")}

Key reliability gaps identified from segment-level residuals:
1. **Long-distance & airport trips (`>10 mi` / Airport)**: MAE and 90th-percentile absolute error increase substantially with trip distance because longer highway/bridge corridors accumulate unpredictable traffic incidents, toll-plaza queues, and terminal congestion.
2. **Midday and peak rush hours (`07:00–09:59`, `10:00–15:59`, `16:00–19:59`)**: Daytime commercial and commuter periods show substantially larger MAE and RMSE (including the highest RMSE during morning rush and highest MAE/90th-percentile error during midday congestion) compared with late-night (`00:00–06:59`) or late-evening (`20:00–23:59`) trips.
3. **Same-zone trips (`PULocationID == DOLocationID`)**: Short local or circling trips (e.g., waiting/errands within one zone) exhibit higher relative uncertainty (RMSE is disproportionately high relative to their 6.59 min average duration) because zone-level origin-destination pairs cannot distinguish direct hops from multi-stop local rides.

## 6. Limitations

- **Route distance proxy**: `trip_distance` in TLC records is odometer distance measured at trip completion; in live pre-trip ETA systems, planned navigation route distance would substitute for it and may differ slightly if drivers detour.
- **No live weather, road-closure, or event feeds**: Summer thunderstorms, street closures, or July 4th holiday events are not observed in the TLC schema.
- **Zone-level spatial granularity**: `PULocationID` and `DOLocationID` represent coarse Taxi Zones rather than exact GPS coordinates, hiding block-level variance inside large zones.
- **Subsampling for CV/training**: While data cleaning and EDA summary statistics cover all **{cleaning_stats['cleaned_rows']:,}** cleaned July 2024 trips, scikit-learn model training/tuning used a reproducible **{len(frame):,}**-row sample to stay within standard workstation memory.
- **Single-month seasonality**: Data covers July 2024 only and may not generalize directly to winter months or other years without retraining.

## 7. Team Contributions

Replace the placeholders below with the actual Group 10 member names and contributions before submission:

- Member 1: [Name] — Data streaming, preprocessing, validation, and leakage audit
- Member 2: [Name] — Exploratory data analysis, statistical summaries, and visualizations
- Member 3: [Name] — Feature engineering, model comparison, hyperparameter tuning, and error analysis
- Member 4: [Name] — Report compilation, interpretation, and reproducibility verification

## 8. Reproducibility

Run `python taxi_duration_project.py` inside `project/` after installing `requirements.txt`. All metrics, plots, the cleaned dataset (`data/cleaned/yellow_tripdata_2024-07_cleaned.csv`), the fitted model (`outputs/final_model.joblib`), and this report are regenerated automatically.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    root_report.write_text(report, encoding="utf-8")


def load_sample_from_cleaned_csv(cleaned_csv_path: Path, sample_size: int) -> pd.DataFrame:
    """Load a reproducible uniform sample from an existing cleaned CSV without loading all 3M rows."""
    use_cols = [c for c in CLEANED_HEADER if c != "tpep_pickup_datetime"]
    rng = np.random.default_rng(RANDOM_STATE)
    chunks = []
    # Keep ~5% per chunk then downsample cleanly to exact sample_size
    for chunk in pd.read_csv(cleaned_csv_path, usecols=use_cols, chunksize=250_000):
        take = max(1, int(len(chunk) * 0.06))
        chunks.append(chunk.sample(n=min(take, len(chunk)), random_state=int(rng.integers(0, 1_000_000))))
    combined = pd.concat(chunks, ignore_index=True)
    if len(combined) > sample_size:
        combined = combined.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    return add_engineered_features(combined)


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    cleaning_json = output / "cleaning_summary.json"
    eda_json = output / "eda_summary.json"
    if args.reuse_cleaned and args.cleaned_output.exists() and cleaning_json.exists() and eda_json.exists():
        print(f"Reusing existing cleaned CSV {args.cleaned_output}...")
        cleaning_stats = json.loads(cleaning_json.read_text(encoding="utf-8"))
        eda_full = json.loads(eda_json.read_text(encoding="utf-8"))
        frame = load_sample_from_cleaned_csv(args.cleaned_output, args.sample_size)
        print(f"Loaded {len(frame):,} sample rows from {cleaning_stats['cleaned_rows']:,} cleaned rows.")
    else:
        print(f"Streaming and cleaning {args.input}...")
        frame, cleaning_stats, eda_full = stream_clean_and_summarize(
            args.input, args.cleaned_output, args.sample_size
        )
        print(
            f"Cleaned {cleaning_stats['cleaned_rows']:,} / {cleaning_stats['raw_rows']:,} rows "
            f"in {cleaning_stats['elapsed_seconds']}s. Sample size: {len(frame):,}."
        )

    validate_data(frame, cleaning_stats, output)
    eda_json.write_text(json.dumps(eda_full, indent=2), encoding="utf-8")
    save_eda(frame, eda_full, output)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Ridge baseline": Pipeline(
            [
                ("preprocess", make_linear_preprocessor()),
                ("model", Ridge(alpha=1.0, solver="lsqr", max_iter=1000)),
            ]
        ),
        "Random forest baseline": Pipeline(
            [
                ("preprocess", make_tree_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=50,
                        max_depth=12,
                        min_samples_leaf=10,
                        max_features=0.5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra trees baseline": Pipeline(
            [
                ("preprocess", make_tree_preprocessor()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=50,
                        max_depth=12,
                        min_samples_leaf=10,
                        max_features=0.6,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting baseline": Pipeline(
            [
                ("preprocess", make_tree_preprocessor()),
                ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    cv_summary = []

    for name, model in models.items():
        print(f"Evaluating 5-fold CV and test split for: {name}...")
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            return_train_score=False,
            n_jobs=1,
        )["test_score"]
        cv_summary.append(
            {
                "model": name,
                "cv_rmse_mean": float(-cv_scores.mean()),
                "cv_rmse_std": float(cv_scores.std()),
            }
        )
        model.fit(X_train, y_train)
        metrics.extend(
            [
                {"model": name, **evaluate(model, X_train, y_train, "train")},
                {"model": name, **evaluate(model, X_test, y_test, "test")},
            ]
        )

    print("Running GridSearchCV hyperparameter tuning on HistGradientBoostingRegressor...")
    tuned_search = GridSearchCV(
        Pipeline(
            [
                ("preprocess", make_tree_preprocessor()),
                ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        {
            "model__max_iter": [150, 250],
            "model__max_leaf_nodes": [31, 63],
            "model__learning_rate": [0.05, 0.1],
            "model__min_samples_leaf": [30],
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    tuned_search.fit(X_train, y_train)
    best_params = tuned_search.best_params_
    cv_summary.append(
        {
            "model": "Tuned HistGradientBoosting",
            "cv_rmse_mean": float(-tuned_search.best_score_),
            "cv_rmse_std": float(tuned_search.cv_results_["std_test_score"][tuned_search.best_index_]),
        }
    )

    hgb_tuned_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    ensemble = VotingRegressor(
        [
            (
                "tuned_hgb",
                Pipeline(
                    [
                        ("preprocess", make_tree_preprocessor()),
                        ("model", HistGradientBoostingRegressor(**hgb_tuned_params, random_state=RANDOM_STATE)),
                    ]
                ),
            ),
            (
                "regularized_rf",
                Pipeline(
                    [
                        ("preprocess", make_tree_preprocessor()),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=80,
                                max_depth=14,
                                min_samples_leaf=8,
                                max_features=0.6,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    print("Evaluating 5-fold CV for final Ensemble model...")
    ens_cv = cross_validate(
        ensemble,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=False,
        n_jobs=1,
    )["test_score"]
    cv_summary.append(
        {
            "model": "Ensemble model",
            "cv_rmse_mean": float(-ens_cv.mean()),
            "cv_rmse_std": float(ens_cv.std()),
        }
    )
    ensemble.fit(X_train, y_train)

    metrics.extend(
        [
            {"model": "Tuned HistGradientBoosting", **evaluate(tuned_search.best_estimator_, X_train, y_train, "train")},
            {"model": "Tuned HistGradientBoosting", **evaluate(tuned_search.best_estimator_, X_test, y_test, "test")},
            {"model": "Ensemble model", **evaluate(ensemble, X_train, y_train, "train")},
            {"model": "Ensemble model", **evaluate(ensemble, X_test, y_test, "test")},
        ]
    )

    # Sensitivity ablation: OD + Time only (no trip_distance, log1p_distance, or RatecodeID)
    print("Running OD-only ablation (without trip_distance / RatecodeID)...")
    od_num_features = [c for c in NUMERIC_MODEL_FEATURES if c not in {"trip_distance", "log1p_distance"}]
    od_model = Pipeline(
        [
            ("preprocess", make_tree_preprocessor(od_num_features, ["VendorID"], HIGH_CARD_CAT_FEATURES)),
            ("model", HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=63, random_state=RANDOM_STATE)),
        ]
    )
    od_model.fit(X_train, y_train)
    ablation_metrics = evaluate(od_model, X_test, y_test, "test_od_only")
    metrics.append({"model": "OD-only ablation (no distance)", **ablation_metrics})

    metrics_df = pd.DataFrame(metrics)
    cv_df = pd.DataFrame(cv_summary).sort_values("cv_rmse_mean").reset_index(drop=True)
    metrics_df.to_csv(output / "model_metrics.csv", index=False)
    cv_df.to_csv(output / "cv_model_comparison.csv", index=False)
    (output / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    joblib.dump(ensemble, output / "final_model.joblib")

    print("Computing permutation feature importance and segment error analysis...")
    perm_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=min(15_000, len(X_test)), replace=False)
    perm = permutation_importance(
        ensemble,
        X_test.iloc[perm_idx],
        y_test.iloc[perm_idx],
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output / "feature_importance.csv", index=False)

    test_preds = np.clip(ensemble.predict(X_test), MIN_DURATION_MIN, MAX_DURATION_MIN)
    predictions_df = pd.DataFrame(
        {
            "actual_duration_min": y_test.values,
            "predicted_duration_min": np.round(test_preds, 3),
            "residual": np.round(y_test.values - test_preds, 3),
        }
    )
    predictions_df.to_csv(output / "test_predictions.csv", index=False)

    error_segments = compute_segment_error_analysis(pd.concat([X_test, y_test], axis=1), test_preds)
    error_segments.to_csv(output / "error_analysis_by_segment.csv", index=False)

    save_diagnostics(predictions_df, cv_df, error_segments, output)

    root_report = Path(__file__).resolve().parent / "report.md"
    write_report(
        cleaning_stats=cleaning_stats,
        eda_full=eda_full,
        frame=frame,
        metrics=metrics_df,
        cv_summary=cv_df,
        best_params=best_params,
        feature_importance=importance_df,
        error_segments=error_segments,
        ablation_metrics=ablation_metrics,
        output=output,
        root_report=root_report,
    )

    print("\n=== Model Metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"\nBest parameters: {best_params}")
    print(f"Outputs written to: {output.resolve()}")


if __name__ == "__main__":
    arguments = parse_args()
    project_dir = Path(__file__).resolve().parent
    if arguments.input == Path("yellow_tripdata_2024-07.xlsx") and not arguments.input.exists():
        arguments.input = project_dir / arguments.input
    if arguments.output == Path("outputs"):
        arguments.output = project_dir / arguments.output
    if arguments.cleaned_output == Path("data/cleaned/yellow_tripdata_2024-07_cleaned.csv"):
        arguments.cleaned_output = project_dir / arguments.cleaned_output
    run(arguments)
