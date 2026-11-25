"""Simple, beginner-friendly NYC Taxi Trip Duration Predictor (Group 10).

Predicts trip duration in minutes using ONLY information available when a trip begins,
combining descriptive statistics (average, median, variance, standard deviation, correlation),
if/else rule-based group averages, simple/multiple linear regression, and Ridge tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "trip_duration_min"

# Post-trip / dropoff / payment columns removed to prevent target leakage
FORBIDDEN_COLUMNS = [
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
]

AIRPORT_ZONES = {1, 132, 138}  # Newark (1), JFK (132), LaGuardia (138)
EXCEL_EPOCH = datetime(1899, 12, 30)
JULY_START_SERIAL = 45474.0  # 2024-07-01 00:00:00
JULY_END_SERIAL = 45505.0    # 2024-08-01 00:00:00

NUMERIC_FEATURES = [
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "passenger_count",
    "is_airport_trip",
    "same_zone",
]
CATEGORICAL_FEATURES = ["VendorID", "RatecodeID"]

CLEANED_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "same_zone",
    "is_airport_trip",
    TARGET,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("yellow_tripdata_2024-07.xlsx"))
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=Path("data/cleaned/yellow_tripdata_2024-07_cleaned.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument(
        "--reuse-cleaned",
        action="store_true",
        help="Reuse existing cleaned CSV instead of re-reading the large .xlsx workbook.",
    )
    return parser.parse_args()


def load_and_clean_xlsx(xlsx_path: Path, cleaned_csv_path: Path, sample_size: int) -> tuple[pd.DataFrame, dict]:
    """Stream all sheets of the raw .xlsx file, remove anomalies and leakage, and save cleaned CSV."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Raw Excel file not found: {xlsx_path}")

    cleaned_csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_re = re.compile(rb"<row [^>]*>(.*?)</row>")
    cell_re = re.compile(rb'<c r="([A-S])\d+"[^>]*>(?:<v>([^<]*)</v>|<is><t>([^<]*)</t></is>)</c>')
    col_idx = {chr(65 + i): i for i in range(19)}

    counts = {"raw_rows": 0, "dropped_invalid": 0, "cleaned_rows": 0}
    rng = np.random.default_rng(RANDOM_STATE)
    sample_rows: list[list] = []

    with (
        zipfile.ZipFile(xlsx_path) as zf,
        cleaned_csv_path.open("w", newline="", encoding="utf-8") as csv_file,
    ):
        writer = csv.writer(csv_file)
        writer.writerow(CLEANED_COLUMNS)

        sheets = sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        for sheet in sheets:
            with zf.open(sheet) as f:
                buf, is_header = b"", True
                while True:
                    chunk = f.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    buf += chunk
                    end = buf.rfind(b"</row>")
                    if end == -1:
                        continue
                    part, buf = buf[: end + 6], buf[end + 6 :]

                    for m in row_re.finditer(part):
                        if is_header:
                            is_header = False
                            continue
                        counts["raw_rows"] += 1
                        cells: list[bytes | None] = [None] * 19
                        for cm in cell_re.finditer(m.group(1)):
                            ci = col_idx.get(chr(cm.group(1)[0]))
                            if ci is not None:
                                cells[ci] = cm.group(2) if cm.group(2) is not None else cm.group(3)
                        try:
                            vendor = int(float(cells[0]))
                            pu_ser = float(cells[1])
                            do_ser = float(cells[2])
                            dist = float(cells[4])
                            pu_zone = int(float(cells[7]))
                            do_zone = int(float(cells[8]))
                        except (TypeError, ValueError):
                            counts["dropped_invalid"] += 1
                            continue

                        dur_min = (do_ser - pu_ser) * 1440.0
                        speed_mph = dist / (dur_min / 60.0) if dur_min > 0 else 0.0

                        # Basic domain validation: July 2024 dates, 1-180 min, 0-80 mi, 1-80 mph, valid zones
                        if (
                            not (JULY_START_SERIAL <= pu_ser < JULY_END_SERIAL)
                            or not (1.0 <= dur_min <= 180.0)
                            or not (0.01 <= dist <= 80.0)
                            or not (1.0 <= speed_mph <= 80.0)
                            or not (1 <= pu_zone <= 265 and 1 <= do_zone <= 265)
                        ):
                            counts["dropped_invalid"] += 1
                            continue

                        pax = np.nan
                        if cells[3]:
                            p_val = int(float(cells[3]))
                            if 1 <= p_val <= 9:
                                pax = p_val
                        ratecode = int(float(cells[5])) if cells[5] and int(float(cells[5])) in {1, 2, 3, 4, 5, 6} else 99

                        dt = EXCEL_EPOCH + timedelta(seconds=round(pu_ser * 86400.0))
                        hour, dow = dt.hour, dt.weekday()
                        is_weekend = 1 if dow >= 5 else 0
                        is_rush = 1 if (dow < 5 and (7 <= hour <= 9 or 16 <= hour <= 19)) else 0
                        is_overnight = 1 if (hour >= 22 or hour < 6) else 0
                        same_zone = 1 if pu_zone == do_zone else 0
                        is_airport = 1 if (pu_zone in AIRPORT_ZONES or do_zone in AIRPORT_ZONES) else 0

                        row = [
                            vendor,
                            dt.strftime("%Y-%m-%d %H:%M:%S"),
                            hour,
                            dow,
                            is_weekend,
                            is_rush,
                            is_overnight,
                            "" if np.isnan(pax) else int(pax),
                            round(dist, 2),
                            ratecode,
                            pu_zone,
                            do_zone,
                            same_zone,
                            is_airport,
                            round(dur_min, 2),
                        ]
                        writer.writerow(row)

                        idx = counts["cleaned_rows"]
                        counts["cleaned_rows"] += 1
                        sample_entry = [row[0], *row[2:7], pax, row[8], *row[9:]]
                        if idx < sample_size:
                            sample_rows.append(sample_entry)
                        else:
                            j = int(rng.integers(0, idx + 1))
                            if j < sample_size:
                                sample_rows[j] = sample_entry

    cols = [c for c in CLEANED_COLUMNS if c != "tpep_pickup_datetime"]
    df = pd.DataFrame(sample_rows, columns=cols)
    return df, counts


def load_sample_from_cleaned_csv(cleaned_csv_path: Path, sample_size: int) -> tuple[pd.DataFrame, dict]:
    """Load a reproducible sample from an existing cleaned CSV without re-parsing .xlsx."""
    use_cols = [c for c in CLEANED_COLUMNS if c != "tpep_pickup_datetime"]
    rng = np.random.default_rng(RANDOM_STATE)
    total_rows = 0
    chunks = []
    for chunk in pd.read_csv(cleaned_csv_path, usecols=use_cols, chunksize=250_000):
        total_rows += len(chunk)
        take = max(1, int(len(chunk) * 0.05))
        chunks.append(chunk.sample(n=min(take, len(chunk)), random_state=int(rng.integers(0, 1_000_000))))
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    return df, {"raw_rows": None, "dropped_invalid": None, "cleaned_rows": total_rows}


def compute_basic_stats(df: pd.DataFrame, counts: dict) -> dict:
    """Calculate basic descriptive statistics: average, median, variance, SD, and correlations."""
    dur = df[TARGET]
    dist = df["trip_distance"]
    corrs = df[NUMERIC_FEATURES].corrwith(dur).round(4).to_dict()

    stats = {
        "cleaning_counts": counts,
        "sample_rows": int(len(df)),
        "duration_stats": {
            "average_mean_min": round(float(dur.mean()), 4),
            "median_min": round(float(dur.median()), 4),
            "variance_min2": round(float(dur.var(ddof=1)), 4),
            "std_dev_min": round(float(dur.std(ddof=1)), 4),
            "min_min": round(float(dur.min()), 4),
            "max_min": round(float(dur.max()), 4),
            "q25_min": round(float(dur.quantile(0.25)), 4),
            "q75_min": round(float(dur.quantile(0.75)), 4),
        },
        "distance_stats": {
            "average_mean_mi": round(float(dist.mean()), 4),
            "median_mi": round(float(dist.median()), 4),
            "variance_mi2": round(float(dist.var(ddof=1)), 4),
            "std_dev_mi": round(float(dist.std(ddof=1)), 4),
        },
        "group_averages_and_sd": {
            "airport_mean_min": round(float(dur[df["is_airport_trip"] == 1].mean()), 3),
            "airport_sd_min": round(float(dur[df["is_airport_trip"] == 1].std(ddof=1)), 3),
            "non_airport_mean_min": round(float(dur[df["is_airport_trip"] == 0].mean()), 3),
            "non_airport_sd_min": round(float(dur[df["is_airport_trip"] == 0].std(ddof=1)), 3),
            "rush_hour_mean_min": round(float(dur[df["is_rush_hour"] == 1].mean()), 3),
            "rush_hour_sd_min": round(float(dur[df["is_rush_hour"] == 1].std(ddof=1)), 3),
            "non_rush_mean_min": round(float(dur[df["is_rush_hour"] == 0].mean()), 3),
            "non_rush_sd_min": round(float(dur[df["is_rush_hour"] == 0].std(ddof=1)), 3),
            "weekend_mean_min": round(float(dur[df["is_weekend"] == 1].mean()), 3),
            "weekday_mean_min": round(float(dur[df["is_weekend"] == 0].mean()), 3),
        },
        "correlations_with_duration": corrs,
    }
    return stats


def if_else_rule_predict(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    """Simple if / elif / else rule-based average predictor fitted on training data only."""
    # Compute group averages on training data using simple conditions
    air_avg = float(train_df.loc[train_df["is_airport_trip"] == 1, TARGET].mean())
    long_avg = float(train_df.loc[(train_df["is_airport_trip"] == 0) & (train_df["trip_distance"] > 5.0), TARGET].mean())
    med_rush_avg = float(
        train_df.loc[
            (train_df["is_airport_trip"] == 0)
            & (train_df["trip_distance"] > 1.5)
            & (train_df["trip_distance"] <= 5.0)
            & (train_df["is_rush_hour"] == 1),
            TARGET,
        ].mean()
    )
    med_off_avg = float(
        train_df.loc[
            (train_df["is_airport_trip"] == 0)
            & (train_df["trip_distance"] > 1.5)
            & (train_df["trip_distance"] <= 5.0)
            & (train_df["is_rush_hour"] == 0),
            TARGET,
        ].mean()
    )
    short_avg = float(train_df.loc[(train_df["is_airport_trip"] == 0) & (train_df["trip_distance"] <= 1.5), TARGET].mean())

    preds = np.empty(len(target_df), dtype=float)
    for i, row in enumerate(target_df.itertuples(index=False)):
        if row.is_airport_trip == 1:
            preds[i] = air_avg
        elif row.trip_distance > 5.0:
            preds[i] = long_avg
        elif row.trip_distance > 1.5:
            if row.is_rush_hour == 1:
                preds[i] = med_rush_avg
            else:
                preds[i] = med_off_avg
        else:
            preds[i] = short_avg
    return preds


def eval_predictions(y_true: pd.Series, y_pred: np.ndarray, split: str) -> dict:
    """Compute MAE, RMSE, and R-squared."""
    y_pred = np.clip(y_pred, 1.0, 180.0)
    return {
        "split": split,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_eda_plots(df: pd.DataFrame, stats: dict, figures_dir: Path) -> None:
    """Generate simple, readable EDA charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    mean_v = stats["duration_stats"]["average_mean_min"]
    sd_v = stats["duration_stats"]["std_dev_min"]
    med_v = stats["duration_stats"]["median_min"]

    # 1. Distribution with Mean, Median, and +/- 1 SD
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df[TARGET].clip(upper=65), bins=50, kde=True, color="#4c72b0")
    plt.axvline(mean_v, color="crimson", linestyle="-", linewidth=2, label=f"Mean ({mean_v:.2f} min)")
    plt.axvline(med_v, color="darkgreen", linestyle="--", linewidth=2, label=f"Median ({med_v:.2f} min)")
    plt.axvline(mean_v + sd_v, color="darkorange", linestyle=":", linewidth=1.8, label=f"Mean + 1 SD ({mean_v + sd_v:.2f})")
    plt.xlabel("Trip Duration (minutes, visual capped at 65)")
    plt.ylabel("Number of Trips")
    plt.title("Trip Duration Distribution (Mean, Median, Standard Deviation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "duration_distribution.png", dpi=150)
    plt.close()

    # 2. Average duration by pickup hour
    hourly = df.groupby("pickup_hour")[TARGET].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8.5, 4.3))
    plt.plot(hourly["pickup_hour"], hourly["mean"], marker="o", color="#2b5c8f", label="Average duration")
    plt.fill_between(
        hourly["pickup_hour"],
        hourly["mean"] - hourly["std"] * 0.5,
        hourly["mean"] + hourly["std"] * 0.5,
        alpha=0.18,
        color="#2b5c8f",
        label="± 0.5 SD band",
    )
    plt.xticks(range(0, 24, 2))
    plt.xlabel("Pickup Hour (0-23)")
    plt.ylabel("Trip Duration (minutes)")
    plt.title("Average Trip Duration by Hour of Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "duration_by_hour.png", dpi=150)
    plt.close()

    # 3. Simple linear trend: Duration vs Distance
    vis = df[df["trip_distance"] <= 25].sample(n=min(5_000, len(df)), random_state=RANDOM_STATE)
    plt.figure(figsize=(8, 4.5))
    sns.regplot(data=vis, x="trip_distance", y=TARGET, scatter_kws={"alpha": 0.25, "s": 14}, line_kws={"color": "crimson"})
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Trip Duration (minutes)")
    plt.title("Trip Duration vs. Distance with Linear Regression Trend")
    plt.tight_layout()
    plt.savefig(figures_dir / "duration_vs_distance.png", dpi=150)
    plt.close()

    # 4. Group averages comparison (Airport, Rush Hour, Weekend)
    grp = stats["group_averages_and_sd"]
    labels = ["Non-Airport", "Airport", "Non-Rush", "Rush Hour", "Weekend", "Weekday"]
    values = [
        grp["non_airport_mean_min"],
        grp["airport_mean_min"],
        grp["non_rush_mean_min"],
        grp["rush_hour_mean_min"],
        grp["weekend_mean_min"],
        grp["weekday_mean_min"],
    ]
    plt.figure(figsize=(8, 4.3))
    sns.barplot(x=labels, y=values, color="#4c72b0")
    plt.ylabel("Average Trip Duration (minutes)")
    plt.title("Average Trip Duration Across Trip-Start Conditions")
    plt.tight_layout()
    plt.savefig(figures_dir / "duration_by_conditions.png", dpi=150)
    plt.close()

    # 5. Correlation heatmap
    corr_cols = ["trip_distance", "is_airport_trip", "is_rush_hour", "is_weekend", "pickup_hour", "passenger_count", "same_zone", TARGET]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation Matrix of Trip-Start Features and Duration")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def save_diagnostic_plots(y_test: pd.Series, preds: np.ndarray, figures_dir: Path) -> None:
    """Save actual-vs-predicted scatter and residual histogram with mean/SD."""
    residuals = y_test.values - preds
    res_mean = float(np.mean(residuals))
    res_sd = float(np.std(residuals, ddof=1))

    idx = np.random.default_rng(RANDOM_STATE).choice(len(y_test), size=min(5_000, len(y_test)), replace=False)
    plt.figure(figsize=(6, 5.5))
    plt.scatter(y_test.values[idx], preds[idx], alpha=0.25, s=14, color="#2b5c8f")
    plt.plot([0, 80], [0, 80], linestyle="--", color="crimson", label="Perfect prediction")
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.xlabel("Actual Duration (minutes)")
    plt.ylabel("Predicted Duration (minutes)")
    plt.title("Actual vs. Predicted Duration (Multiple Linear Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7.5, 4.2))
    sns.histplot(np.clip(residuals, -25, 25), bins=50, kde=True, color="#4c72b0")
    plt.axvline(res_mean, color="crimson", linestyle="--", label=f"Mean residual = {res_mean:.2f} min")
    plt.axvline(res_sd, color="darkorange", linestyle=":", label=f"+1 Residual SD = {res_sd:.2f} min")
    plt.xlabel("Residual = Actual - Predicted (minutes)")
    plt.title("Prediction Residual Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "residual_distribution.png", dpi=150)
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    """Median imputation + standardization for numeric inputs and one-hot encoding for VendorID/RatecodeID."""
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", num_pipe, NUMERIC_FEATURES), ("cat", cat_pipe, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def compute_segment_reliability(test_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    """Measure MAE, RMSE, and duration SD across simple segments to identify less reliable predictions."""
    df = test_df.copy()
    df["actual"] = df[TARGET]
    df["pred"] = np.clip(preds, 1.0, 180.0)
    df["abs_err"] = np.abs(df["actual"] - df["pred"])
    df["sq_err"] = (df["actual"] - df["pred"]) ** 2

    df["Distance Group"] = pd.cut(
        df["trip_distance"],
        bins=[0, 1.5, 5.0, 10.0, 100.0],
        labels=["Short (<=1.5 mi)", "Medium (1.5-5 mi)", "Long (5-10 mi)", "Very Long (>10 mi)"],
    ).astype(str)
    df["Rush Hour"] = np.where(df["is_rush_hour"] == 1, "Rush Hour", "Non-Rush Hour")
    df["Airport"] = np.where(df["is_airport_trip"] == 1, "Airport Trip", "Non-Airport Trip")

    rows = []
    for dim in ["Distance Group", "Rush Hour", "Airport"]:
        for seg, grp in df.groupby(dim, observed=False):
            rows.append(
                {
                    "dimension": dim,
                    "segment": str(seg),
                    "trips": int(len(grp)),
                    "mean_duration_min": round(float(grp["actual"].mean()), 3),
                    "sd_duration_min": round(float(grp["actual"].std(ddof=1)), 3),
                    "mae": round(float(grp["abs_err"].mean()), 3),
                    "rmse": round(float(np.sqrt(grp["sq_err"].mean())), 3),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    stats: dict,
    metrics_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    best_alpha: float,
    simple_slope: float,
    simple_intercept: float,
    segment_df: pd.DataFrame,
    root_report: Path,
) -> None:
    """Generate a clear, concise report.md summarizing statistics, models, findings, and limitations."""
    dur = stats["duration_stats"]
    dist = stats["distance_stats"]
    grp = stats["group_averages_and_sd"]
    best_row = metrics_df[(metrics_df["model"] == "Tuned Ridge Regression") & (metrics_df["split"] == "test")].iloc[0]
    avg_row = metrics_df[(metrics_df["model"] == "Overall Average Baseline") & (metrics_df["split"] == "test")].iloc[0]
    rule_row = metrics_df[(metrics_df["model"] == "If/Else Group-Average Rules") & (metrics_df["split"] == "test")].iloc[0]

    report = f"""# Group 10 — Simple Taxi Trip Duration Prediction Report

## 1. Objective and Data
This project investigates patterns in NYC Yellow Taxi trip duration using basic statistical analysis and builds a simple, interpretable predictive model for **trip duration in minutes (`{TARGET}`)** using **only information available when a trip begins**.

- **Raw dataset**: `yellow_tripdata_2024-07.xlsx` (3 sheets)
- **Cleaned rows**: **{stats['cleaning_counts']['cleaned_rows']:,}** valid July 2024 trips (`1–180` min, `0.01–80` miles, `1–80` mph, valid NYC zone IDs `1–265`)
- **Analysis sample**: **{stats['sample_rows']:,}** uniform random trips split **80% train ({int(stats['sample_rows']*0.8):,}) / 20% test ({int(stats['sample_rows']*0.2):,})**
- **Strict leakage removal**: Dropped all 12 post-trip columns (`{", ".join(FORBIDDEN_COLUMNS)}`)

## 2. Basic Statistical Summary (Average, Median, Variance, SD, Correlation)

| Variable | Mean (Average) | Median | Variance | Standard Deviation (SD) |
|---|---:|---:|---:|---:|
| `trip_duration_min` | {dur['average_mean_min']:.2f} min | {dur['median_min']:.2f} min | {dur['variance_min2']:.2f} | {dur['std_dev_min']:.2f} min |
| `trip_distance` | {dist['average_mean_mi']:.2f} mi | {dist['median_mi']:.2f} mi | {dist['variance_mi2']:.2f} | {dist['std_dev_mi']:.2f} mi |

### Group Averages & Standard Deviations
- **Airport vs. Non-Airport**: Airport trips average **{grp['airport_mean_min']:.2f} min** (SD = {grp['airport_sd_min']:.2f} min) vs. **{grp['non_airport_mean_min']:.2f} min** (SD = {grp['non_airport_sd_min']:.2f} min) for non-airport trips.
- **Rush Hour vs. Non-Rush**: Weekday rush-hour trips (`07–09`, `16–19`) average **{grp['rush_hour_mean_min']:.2f} min** (SD = {grp['rush_hour_sd_min']:.2f} min) vs. **{grp['non_rush_mean_min']:.2f} min** (SD = {grp['non_rush_sd_min']:.2f} min).
- **Weekday vs. Weekend**: Weekdays average **{grp['weekday_mean_min']:.2f} min** vs. **{grp['weekend_mean_min']:.2f} min** on weekends.
- **Strongest Correlation with Duration**: `trip_distance` ($r = {stats['correlations_with_duration']['trip_distance']:.4f}$) and `is_airport_trip` ($r = {stats['correlations_with_duration']['is_airport_trip']:.4f}$).

## 3. Modeling Approach & Hyperparameter Tuning

We progress from simple statistical baselines to linear regression:
1. **Overall Average Baseline**: Predicts the training mean (`{dur['average_mean_min']:.2f} min`) for every trip.
2. **If/Else Group-Average Rules**: Assigns the training group average using explicit `if / elif / else` rules on airport status, trip distance (`>5 mi`, `1.5–5 mi`, `<=1.5 mi`), and rush-hour flag.
3. **Simple Linear Regression**: Fits $\\text{{duration}} = {simple_intercept:.3f} + {simple_slope:.3f} \\times \\text{{trip\\_distance}}$.
4. **Multiple Linear Regression**: Uses `trip_distance`, time-of-day flags (`pickup_hour`, `pickup_dayofweek`, `is_weekend`, `is_rush_hour`, `is_overnight`), `passenger_count`, `is_airport_trip`, `same_zone`, `VendorID`, and `RatecodeID` with median imputation and standardization.
5. **Tuned Ridge Linear Regression**: Tunes L2 regularization `alpha` via 5-fold `KFold` `GridSearchCV` over `[0.01, 0.1, 1.0, 10.0, 100.0]`. Selected `alpha = {best_alpha}`.

### Ridge 5-Fold Cross-Validation Tuning

{cv_df.to_markdown(index=False)}

### Train & Untouched Test Results

{metrics_df.to_markdown(index=False)}

- **Takeaway**: Predicting the overall average gives RMSE **{avg_row['rmse']:.3f} min** ($R^2 = {avg_row['r2']:.3f}$). Simple `if/else` group averages cut RMSE to **{rule_row['rmse']:.3f} min** ($R^2 = {rule_row['r2']:.3f}$). Linear Regression and Tuned Ridge Regression (`alpha={best_alpha}`) improve accuracy further to **MAE = {best_row['mae']:.3f} min**, **RMSE = {best_row['rmse']:.3f} min**, and **$R^2 = {best_row['r2']:.3f}$**.

## 4. Travel-Time Estimation & Where Predictions Are Less Reliable

{segment_df.to_markdown(index=False)}

- **High-variance / less reliable situations**: Longer trips (`>10 mi`, SD = {segment_df.loc[segment_df['segment']=='Very Long (>10 mi)', 'sd_duration_min'].values[0]:.2f} min, MAE = {segment_df.loc[segment_df['segment']=='Very Long (>10 mi)', 'mae'].values[0]:.2f} min), airport trips, and peak rush hours exhibit much wider travel-time spread (higher variance/SD) due to highway congestion, toll queues, and traffic incidents.
- **How findings help**: Knowing only planned distance, pickup time, and whether a trip is an airport/rush-hour ride provides a fast, transparent baseline travel-time estimate before a trip starts.

## 5. Limitations
- Uses linear relationships only, so non-linear congestion curves and street-level bottlenecks are not modeled.
- `trip_distance` in historical TLC logs is recorded at trip completion; in live ETA systems, planned route distance is used as a pre-trip proxy.
- Weather, road closures, and holiday events (e.g., July 4th) are not included in the dataset.

## 6. Team Contributions
- Member 1: [Name] — Data cleaning, leakage removal, and validation
- Member 2: [Name] — Descriptive statistics (mean, variance, SD, correlations) and EDA plots
- Member 3: [Name] — If/else baseline, linear regression modeling, and Ridge hyperparameter tuning
- Member 4: [Name] — Reliability analysis, README, and final report
"""
    root_report.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / "figures"

    if args.reuse_cleaned and args.cleaned_output.exists():
        print(f"Loading sample from existing cleaned CSV: {args.cleaned_output}")
        df, counts = load_sample_from_cleaned_csv(args.cleaned_output, args.sample_size)
    else:
        print(f"Cleaning raw dataset: {args.input}")
        df, counts = load_and_clean_xlsx(args.input, args.cleaned_output, args.sample_size)

    stats = compute_basic_stats(df, counts)
    (args.output / "basic_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    save_eda_plots(df, stats, figures_dir)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    train_full = pd.concat([X_train, y_train], axis=1)

    metrics = []

    # 1. Overall Average Baseline
    mean_pred_val = float(y_train.mean())
    metrics.append({"model": "Overall Average Baseline", **eval_predictions(y_train, np.full(len(y_train), mean_pred_val), "train")})
    metrics.append({"model": "Overall Average Baseline", **eval_predictions(y_test, np.full(len(y_test), mean_pred_val), "test")})

    # 2. If/Else Group-Average Rules
    metrics.append({"model": "If/Else Group-Average Rules", **eval_predictions(y_train, if_else_rule_predict(train_full, X_train), "train")})
    metrics.append({"model": "If/Else Group-Average Rules", **eval_predictions(y_test, if_else_rule_predict(train_full, X_test), "test")})

    # 3. Simple Linear Regression (trip_distance only)
    simple_lr = LinearRegression().fit(X_train[["trip_distance"]], y_train)
    simple_slope = float(simple_lr.coef_[0])
    simple_intercept = float(simple_lr.intercept_)
    metrics.append({"model": "Simple Linear Regression (Distance)", **eval_predictions(y_train, simple_lr.predict(X_train[["trip_distance"]]), "train")})
    metrics.append({"model": "Simple Linear Regression (Distance)", **eval_predictions(y_test, simple_lr.predict(X_test[["trip_distance"]]), "test")})

    # 4. Multiple Linear Regression (all trip-start features)
    mlr = Pipeline([("preprocess", build_preprocessor()), ("model", LinearRegression())])
    mlr.fit(X_train, y_train)
    mlr_test_preds = mlr.predict(X_test)
    metrics.append({"model": "Multiple Linear Regression", **eval_predictions(y_train, mlr.predict(X_train), "train")})
    metrics.append({"model": "Multiple Linear Regression", **eval_predictions(y_test, mlr_test_preds, "test")})

    # 5. Hyperparameter Tuning: Ridge Linear Regression (alpha tuning via 5-fold CV)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ridge_search = GridSearchCV(
        Pipeline([("preprocess", build_preprocessor()), ("model", Ridge())]),
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=cv,
        scoring="neg_root_mean_squared_error",
        refit=True,
    )
    ridge_search.fit(X_train, y_train)
    best_alpha = float(ridge_search.best_params_["model__alpha"])
    ridge_test_preds = ridge_search.predict(X_test)
    metrics.append({"model": "Tuned Ridge Regression", **eval_predictions(y_train, ridge_search.predict(X_train), "train")})
    metrics.append({"model": "Tuned Ridge Regression", **eval_predictions(y_test, ridge_test_preds, "test")})

    cv_df = pd.DataFrame(
        {
            "alpha": ridge_search.cv_results_["param_model__alpha"],
            "cv_rmse_mean": np.round(-ridge_search.cv_results_["mean_test_score"], 4),
            "cv_rmse_std": np.round(ridge_search.cv_results_["std_test_score"], 4),
        }
    )
    cv_df.to_csv(args.output / "cv_tuning.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output / "model_metrics.csv", index=False)

    # Save linear model coefficients
    feat_names = mlr.named_steps["preprocess"].get_feature_names_out()
    coef_df = pd.DataFrame(
        {"feature": feat_names, "coefficient": np.round(mlr.named_steps["model"].coef_, 4)}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coef_df.to_csv(args.output / "linear_coefficients.csv", index=False)

    # Save test predictions & segment reliability analysis
    pd.DataFrame(
        {
            "actual_duration_min": y_test.values,
            "predicted_duration_min": np.round(np.clip(ridge_test_preds, 1.0, 180.0), 3),
        }
    ).to_csv(args.output / "test_predictions.csv", index=False)

    segment_df = compute_segment_reliability(pd.concat([X_test, y_test], axis=1), ridge_test_preds)
    segment_df.to_csv(args.output / "segment_errors.csv", index=False)
    save_diagnostic_plots(y_test, ridge_test_preds, figures_dir)

    root_report = Path(__file__).resolve().parent / "report.md"
    write_report(stats, metrics_df, cv_df, best_alpha, simple_slope, simple_intercept, segment_df, root_report)

    print("\n=== Basic Statistics ===")
    print(json.dumps(stats["duration_stats"], indent=2))
    print("\n=== Model Metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSimple Linear Regression: duration = {simple_intercept:.3f} + {simple_slope:.3f} * trip_distance")
    print(f"Best Ridge alpha: {best_alpha}")


if __name__ == "__main__":
    main()
