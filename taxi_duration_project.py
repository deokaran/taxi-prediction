"""Simple NYC Taxi Trip Duration Predictor (Group 10).

Predicts trip duration in minutes using ONLY trip-start information,
basic statistics (mean, median, variance, SD, correlation), if/else rules,
simple/multiple linear regression, and Ridge alpha tuning.
"""

from __future__ import annotations

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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 42
TARGET = "trip_duration_min"
FEATURES = [
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
AIRPORT_ZONES = {1, 132, 138}  # Newark (1), JFK (132), LaGuardia (138)
EXCEL_EPOCH = datetime(1899, 12, 30)

BASE_DIR = Path(__file__).resolve().parent
RAW_XLSX = BASE_DIR / "yellow_tripdata_2024-07.xlsx"
CLEANED_CSV = BASE_DIR / "data/cleaned/yellow_tripdata_2024-07_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def build_cleaned_csv(xlsx_path: Path, csv_path: Path) -> None:
    """Read raw 3-sheet .xlsx, drop post-trip leakage & invalid rows, and save cleaned CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_re = re.compile(rb"<row [^>]*>(.*?)</row>")
    cell_re = re.compile(rb'<c r="([A-S])\d+"[^>]*>(?:<v>([^<]*)</v>|<is><t>([^<]*)</t></is>)</c>')
    col_idx = {chr(65 + i): i for i in range(19)}
    header = ["tpep_pickup_datetime", *FEATURES, "PULocationID", "DOLocationID", TARGET]

    with zipfile.ZipFile(xlsx_path) as zf, csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        sheets = sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        for sheet in sheets:
            with zf.open(sheet) as f:
                buf, is_header = b"", True
                while chunk := f.read(4 * 1024 * 1024):
                    buf += chunk
                    end = buf.rfind(b"</row>")
                    if end == -1:
                        continue
                    part, buf = buf[: end + 6], buf[end + 6 :]
                    for m in row_re.finditer(part):
                        if is_header:
                            is_header = False
                            continue
                        cells: list[bytes | None] = [None] * 19
                        for cm in cell_re.finditer(m.group(1)):
                            ci = col_idx.get(chr(cm.group(1)[0]))
                            if ci is not None:
                                cells[ci] = cm.group(2) if cm.group(2) is not None else cm.group(3)
                        try:
                            pu_ser, do_ser = float(cells[1]), float(cells[2])
                            dist = float(cells[4])
                            pu_zone, do_zone = int(float(cells[7])), int(float(cells[8]))
                        except (TypeError, ValueError):
                            continue

                        dur_min = (do_ser - pu_ser) * 1440.0
                        speed_mph = dist / (dur_min / 60.0) if dur_min > 0 else 0.0
                        # Valid July 2024 trip bounds: 1-180 min, 0.01-80 mi, 1-80 mph, zones 1-265
                        if not (
                            45474.0 <= pu_ser < 45505.0
                            and 1.0 <= dur_min <= 180.0
                            and 0.01 <= dist <= 80.0
                            and 1.0 <= speed_mph <= 80.0
                            and 1 <= pu_zone <= 265
                            and 1 <= do_zone <= 265
                        ):
                            continue

                        pax = ""
                        if cells[3]:
                            p_val = int(float(cells[3]))
                            if 1 <= p_val <= 9:
                                pax = p_val

                        dt = EXCEL_EPOCH + timedelta(seconds=round(pu_ser * 86400.0))
                        hour, dow = dt.hour, dt.weekday()
                        is_weekend = int(dow >= 5)
                        is_rush = int(dow < 5 and (7 <= hour <= 9 or 16 <= hour <= 19))
                        is_overnight = int(hour >= 22 or hour < 6)
                        is_airport = int(pu_zone in AIRPORT_ZONES or do_zone in AIRPORT_ZONES)
                        same_zone = int(pu_zone == do_zone)

                        writer.writerow([
                            dt.strftime("%Y-%m-%d %H:%M:%S"),
                            round(dist, 2), hour, dow, is_weekend, is_rush, is_overnight,
                            pax, is_airport, same_zone, pu_zone, do_zone, round(dur_min, 2),
                        ])


def load_sample(sample_size: int = 100_000) -> tuple[pd.DataFrame, int]:
    """Load a reproducible sample from cleaned CSV (building it from .xlsx if missing)."""
    if not CLEANED_CSV.exists():
        print(f"Building cleaned CSV from {RAW_XLSX.name} (one-time setup)...")
        build_cleaned_csv(RAW_XLSX, CLEANED_CSV)

    total_rows, chunks = 0, []
    for chunk in pd.read_csv(CLEANED_CSV, usecols=[*FEATURES, TARGET], chunksize=250_000):
        total_rows += len(chunk)
        chunks.append(chunk.sample(frac=0.05, random_state=SEED))
    df = pd.concat(chunks, ignore_index=True).sample(n=sample_size, random_state=SEED).reset_index(drop=True)
    return df, total_rows


def predict_with_if_else(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    """Simple if/elif/else rule-based predictor using training group averages."""
    air_avg = train_df.loc[train_df["is_airport_trip"] == 1, TARGET].mean()
    long_avg = train_df.loc[(train_df["is_airport_trip"] == 0) & (train_df["trip_distance"] > 5), TARGET].mean()
    rush_avg = train_df.loc[(train_df["is_airport_trip"] == 0) & (train_df["trip_distance"].between(1.5, 5)) & (train_df["is_rush_hour"] == 1), TARGET].mean()
    off_avg = train_df.loc[(train_df["is_airport_trip"] == 0) & (train_df["trip_distance"].between(1.5, 5)) & (train_df["is_rush_hour"] == 0), TARGET].mean()
    short_avg = train_df.loc[(train_df["is_airport_trip"] == 0) & (train_df["trip_distance"] < 1.5), TARGET].mean()

    preds = []
    for row in target_df.itertuples(index=False):
        if row.is_airport_trip == 1:
            preds.append(air_avg)
        elif row.trip_distance > 5:
            preds.append(long_avg)
        elif row.trip_distance >= 1.5:
            preds.append(rush_avg if row.is_rush_hour == 1 else off_avg)
        else:
            preds.append(short_avg)
    return np.array(preds)


def score(y_true: pd.Series, y_pred: np.ndarray, model_name: str, split: str) -> dict:
    """Compute MAE, RMSE, and R2."""
    y_pred = np.clip(y_pred, 1.0, 180.0)
    return {
        "model": model_name,
        "split": split,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def make_plots(df: pd.DataFrame, stats: dict, y_test: pd.Series, preds: np.ndarray) -> None:
    """Create clean EDA and model diagnostic charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    mean_v, med_v, sd_v = stats["mean_min"], stats["median_min"], stats["sd_min"]

    # 1. Duration distribution with Mean, Median, and SD
    plt.figure(figsize=(8, 4.3))
    sns.histplot(df[TARGET].clip(upper=65), bins=50, kde=True, color="#4c72b0")
    plt.axvline(mean_v, color="crimson", linestyle="-", label=f"Mean ({mean_v:.2f} min)")
    plt.axvline(med_v, color="darkgreen", linestyle="--", label=f"Median ({med_v:.2f} min)")
    plt.axvline(mean_v + sd_v, color="darkorange", linestyle=":", label=f"Mean + 1 SD ({mean_v + sd_v:.2f})")
    plt.title("Trip Duration Distribution (Mean, Median, Standard Deviation)")
    plt.xlabel("Trip Duration (minutes, capped at 65)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duration_distribution.png", dpi=150)
    plt.close()

    # 2. Average duration by hour
    hourly = df.groupby("pickup_hour")[TARGET].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 4.2))
    plt.plot(hourly["pickup_hour"], hourly["mean"], marker="o", color="#2b5c8f", label="Average")
    plt.fill_between(hourly["pickup_hour"], hourly["mean"] - 0.5 * hourly["std"], hourly["mean"] + 0.5 * hourly["std"], alpha=0.18)
    plt.title("Average Trip Duration by Pickup Hour")
    plt.xlabel("Pickup Hour (0-23)")
    plt.ylabel("Duration (minutes)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duration_by_hour.png", dpi=150)
    plt.close()

    # 3. Distance vs Duration linear trend
    vis = df[df["trip_distance"] <= 25].sample(n=5_000, random_state=SEED)
    plt.figure(figsize=(8, 4.3))
    sns.regplot(data=vis, x="trip_distance", y=TARGET, scatter_kws={"alpha": 0.25, "s": 14}, line_kws={"color": "crimson"})
    plt.title("Trip Duration vs. Distance (Linear Regression Trend)")
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Duration (minutes)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duration_vs_distance.png", dpi=150)
    plt.close()

    # 4. Group averages comparison
    grp = stats["group_averages"]
    plt.figure(figsize=(7.5, 4))
    sns.barplot(x=list(grp.keys()), y=list(grp.values()), color="#4c72b0")
    plt.xticks(rotation=15)
    plt.ylabel("Average Duration (minutes)")
    plt.title("Average Trip Duration by Trip-Start Condition")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duration_by_conditions.png", dpi=150)
    plt.close()

    # 5. Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[[*FEATURES, TARGET]].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation Matrix (Trip-Start Features vs. Duration)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 6. Actual vs Predicted & 7. Residual Distribution
    idx = np.random.default_rng(SEED).choice(len(y_test), size=5_000, replace=False)
    plt.figure(figsize=(6, 5.5))
    plt.scatter(y_test.values[idx], preds[idx], alpha=0.25, s=14, color="#2b5c8f")
    plt.plot([0, 80], [0, 80], linestyle="--", color="crimson", label="Perfect fit")
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.xlabel("Actual Duration (min)")
    plt.ylabel("Predicted Duration (min)")
    plt.title("Actual vs. Predicted Duration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    residuals = y_test.values - preds
    plt.figure(figsize=(7.5, 4))
    sns.histplot(np.clip(residuals, -25, 25), bins=50, kde=True, color="#4c72b0")
    plt.axvline(np.mean(residuals), color="crimson", linestyle="--", label=f"Mean = {np.mean(residuals):.2f}")
    plt.axvline(np.std(residuals, ddof=1), color="darkorange", linestyle=":", label=f"SD = {np.std(residuals, ddof=1):.2f}")
    plt.title("Residual Distribution (Actual - Predicted)")
    plt.xlabel("Residual (minutes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, total_cleaned_rows = load_sample(100_000)

    # 1. Descriptive statistics: mean, median, variance, SD, correlations
    dur = df[TARGET]
    stats = {
        "cleaned_rows": total_cleaned_rows,
        "sample_rows": len(df),
        "mean_min": round(float(dur.mean()), 4),
        "median_min": round(float(dur.median()), 4),
        "variance_min2": round(float(dur.var(ddof=1)), 4),
        "sd_min": round(float(dur.std(ddof=1)), 4),
        "distance_mean_mi": round(float(df["trip_distance"].mean()), 4),
        "distance_var_mi2": round(float(df["trip_distance"].var(ddof=1)), 4),
        "distance_sd_mi": round(float(df["trip_distance"].std(ddof=1)), 4),
        "group_averages": {
            "Non-Airport": round(float(dur[df["is_airport_trip"] == 0].mean()), 2),
            "Airport": round(float(dur[df["is_airport_trip"] == 1].mean()), 2),
            "Non-Rush": round(float(dur[df["is_rush_hour"] == 0].mean()), 2),
            "Rush Hour": round(float(dur[df["is_rush_hour"] == 1].mean()), 2),
            "Weekend": round(float(dur[df["is_weekend"] == 1].mean()), 2),
            "Weekday": round(float(dur[df["is_weekend"] == 0].mean()), 2),
        },
        "correlations": df[FEATURES].corrwith(dur).round(4).to_dict(),
    }
    (OUTPUT_DIR / "basic_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # 2. Train / test split (80/20) + simple median imputation on train only
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], dur, test_size=0.2, random_state=SEED)
    pax_median = float(X_train["passenger_count"].median())
    X_train = X_train.fillna({"passenger_count": pax_median})
    X_test = X_test.fillna({"passenger_count": pax_median})
    train_full = pd.concat([X_train, y_train], axis=1)

    results = []
    # Model 1: Overall Average
    avg_val = float(y_train.mean())
    results.append(score(y_train, np.full(len(y_train), avg_val), "Overall Average Baseline", "train"))
    results.append(score(y_test, np.full(len(y_test), avg_val), "Overall Average Baseline", "test"))

    # Model 2: If/Else Group-Average Rules
    results.append(score(y_train, predict_with_if_else(train_full, X_train), "If/Else Group-Average Rules", "train"))
    results.append(score(y_test, predict_with_if_else(train_full, X_test), "If/Else Group-Average Rules", "test"))

    # Model 3: Simple Linear Regression (distance only)
    slr = LinearRegression().fit(X_train[["trip_distance"]], y_train)
    results.append(score(y_train, slr.predict(X_train[["trip_distance"]]), "Simple Linear Regression (Distance)", "train"))
    results.append(score(y_test, slr.predict(X_test[["trip_distance"]]), "Simple Linear Regression (Distance)", "test"))

    # Model 4: Multiple Linear Regression (all trip-start features)
    mlr = LinearRegression().fit(X_train, y_train)
    results.append(score(y_train, mlr.predict(X_train), "Multiple Linear Regression", "train"))
    results.append(score(y_test, mlr.predict(X_test), "Multiple Linear Regression", "test"))

    # Model 5: Hyperparameter tuning for Ridge Regression via 5-fold CV
    cv_rows = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        cv_rmse = -cross_val_score(Ridge(alpha=alpha), X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        cv_rows.append({"alpha": alpha, "cv_rmse_mean": round(float(cv_rmse.mean()), 4), "cv_rmse_std": round(float(cv_rmse.std()), 4)})
    cv_df = pd.DataFrame(cv_rows)
    best_alpha = float(cv_df.loc[cv_df["cv_rmse_mean"].idxmin(), "alpha"])

    ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
    ridge_preds = np.clip(ridge.predict(X_test), 1.0, 180.0)
    results.append(score(y_train, ridge.predict(X_train), "Tuned Ridge Regression", "train"))
    results.append(score(y_test, ridge_preds, "Tuned Ridge Regression", "test"))

    # 3. Save tables, segment reliability, plots, and report.md
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "cv_tuning.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "coefficient": np.round(mlr.coef_, 4)}).to_csv(OUTPUT_DIR / "linear_coefficients.csv", index=False)
    pd.DataFrame({"actual_duration_min": y_test.values, "predicted_duration_min": np.round(ridge_preds, 3)}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    seg_df = X_test.assign(actual=y_test.values, err=np.abs(y_test.values - ridge_preds), sq_err=(y_test.values - ridge_preds) ** 2)
    seg_df["Distance Group"] = pd.cut(seg_df["trip_distance"], [0, 1.5, 5, 10, 100], labels=["Short (<=1.5 mi)", "Medium (1.5-5 mi)", "Long (5-10 mi)", "Very Long (>10 mi)"]).astype(str)
    seg_df["Rush Hour"] = np.where(seg_df["is_rush_hour"] == 1, "Rush Hour", "Non-Rush Hour")
    seg_df["Airport"] = np.where(seg_df["is_airport_trip"] == 1, "Airport Trip", "Non-Airport Trip")

    seg_rows = []
    for dim in ["Distance Group", "Rush Hour", "Airport"]:
        for seg, g in seg_df.groupby(dim, observed=False):
            seg_rows.append({"dimension": dim, "segment": seg, "trips": len(g), "mean_min": round(g["actual"].mean(), 2), "sd_min": round(g["actual"].std(ddof=1), 2), "mae": round(g["err"].mean(), 2), "rmse": round(np.sqrt(g["sq_err"].mean()), 2)})
    segment_df = pd.DataFrame(seg_rows)
    segment_df.to_csv(OUTPUT_DIR / "segment_errors.csv", index=False)

    make_plots(df, stats, y_test, ridge_preds)

    best_row = metrics_df[(metrics_df["model"] == "Tuned Ridge Regression") & (metrics_df["split"] == "test")].iloc[0]
    report = f"""# Group 10 — Simple Taxi Trip Duration Prediction Report

## 1. Objective & Data
Predict NYC Yellow Taxi trip duration (`{TARGET}`, minutes) using **only information available when the trip begins**.
- **Cleaned dataset**: **{total_cleaned_rows:,}** valid July 2024 trips (`100,000`-trip sample split 80/20 into 80,000 train / 20,000 test).
- **Leakage columns removed**: `tpep_dropoff_datetime`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `payment_type`, `store_and_fwd_flag`.

## 2. Basic Statistics (Average, Median, Variance, SD, Correlation)
- **Trip duration**: Mean = **{stats['mean_min']:.2f} min**, Median = **{stats['median_min']:.2f} min**, Variance = **{stats['variance_min2']:.2f}**, SD = **{stats['sd_min']:.2f} min**
- **Trip distance**: Mean = **{stats['distance_mean_mi']:.2f} mi**, Variance = **{stats['distance_var_mi2']:.2f}**, SD = **{stats['distance_sd_mi']:.2f} mi**
- **Group averages**: Airport (**{stats['group_averages']['Airport']:.2f} min**) vs. Non-Airport (**{stats['group_averages']['Non-Airport']:.2f} min**); Rush Hour (**{stats['group_averages']['Rush Hour']:.2f} min**) vs. Non-Rush (**{stats['group_averages']['Non-Rush']:.2f} min**)
- **Strongest correlations**: `trip_distance` ($r = {stats['correlations']['trip_distance']:.4f}$), `is_airport_trip` ($r = {stats['correlations']['is_airport_trip']:.4f}$)

## 3. Models & Ridge Alpha Tuning
Simple Linear Regression formula: $\\text{{duration}} = {slr.intercept_:.3f} + {slr.coef_[0]:.3f} \\times \\text{{trip\\_distance}}$.

### 5-Fold Cross-Validation (`Ridge` alpha tuning)
{cv_df.to_markdown(index=False)}

### Train & Untouched Test Results
{metrics_df.to_markdown(index=False)}

Final Tuned Ridge Regression (`alpha = {best_alpha}`): **MAE = {best_row['mae']:.3f} min**, **RMSE = {best_row['rmse']:.3f} min**, **$R^2$ = {best_row['r2']:.3f}**.

## 4. Where Predictions Are Less Reliable
{segment_df.to_markdown(index=False)}

Long trips (`>10 mi`), airport trips, and rush-hour rides have the largest duration SD (highest variance), causing larger prediction errors.

## 5. Limitations & Team Contributions
- **Limitations**: Linear models do not capture sharp non-linear traffic bottlenecks; historical `trip_distance` proxies pre-trip planned route distance; weather and events are unobserved.
- **Team**: Member 1 [Name] (Cleaning) | Member 2 [Name] (Stats & EDA) | Member 3 [Name] (If/Else & Linear Regression) | Member 4 [Name] (Report & Review)
"""
    (BASE_DIR / "report.md").write_text(report, encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"\nSimple Linear Regression: duration = {slr.intercept_:.3f} + {slr.coef_[0]:.3f} * trip_distance")
    print(f"Best Ridge alpha: {best_alpha}")


if __name__ == "__main__":
    main()
