# Simple NYC Taxi Trip Duration Predictor (Group 10)

> **A clean, single-file Python project predicting NYC Yellow Taxi trip duration in minutes using basic statistics (mean, median, variance, standard deviation, correlation), `if/else` group averages, and linear regression—strictly from trip-start information.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearRegression%20%7C%20Ridge-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/status-submission--ready-success)

## Project overview

This project examines NYC TLC Yellow Taxi trip records (`yellow_tripdata_2024-07.xlsx`) to understand how trip duration varies with distance, time of day, rush hours, weekends, and airport zones, and builds a simple, interpretable regression model using **only information available when a trip begins**.

Post-trip fields (`tpep_dropoff_datetime`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `payment_type`, `store_and_fwd_flag`) are dropped before modeling to prevent target leakage.

## Project structure

```text
project/
|-- yellow_tripdata_2024-07.xlsx                  # Raw 3-sheet NYC TLC workbook (3,076,903 rows, ignored in git)
|-- data/
|   `-- cleaned/
|       `-- yellow_tripdata_2024-07_cleaned.csv   # Full cleaned CSV (3,010,079 rows, ignored in git)
|-- outputs/
|   |-- figures/
|   |   |-- actual_vs_predicted.png
|   |   |-- correlation_heatmap.png
|   |   |-- duration_by_conditions.png
|   |   |-- duration_by_hour.png
|   |   |-- duration_distribution.png
|   |   |-- duration_vs_distance.png
|   |   `-- residual_distribution.png
|   |-- basic_stats.json
|   |-- cv_tuning.csv
|   |-- linear_coefficients.csv
|   |-- model_metrics.csv
|   |-- segment_errors.csv
|   `-- test_predictions.csv
|-- taxi_duration_project.py                      # Single-file Python script
|-- report.md                                     # Project report
`-- README.md                                     # Instructions & summary
```

## How to run

From inside `project/`:

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install matplotlib numpy pandas scikit-learn seaborn tabulate
python taxi_duration_project.py
```

If `data/cleaned/yellow_tripdata_2024-07_cleaned.csv` has already been created and you want to re-run statistics, modeling, and plots in a few seconds:

```bash
python taxi_duration_project.py --reuse-cleaned
```

## Statistical & modeling pipeline

1. **Data cleaning**: Streams all 3 sheets of `yellow_tripdata_2024-07.xlsx` (`3,076,903` raw rows), retains **`3,010,079`** valid July 2024 trips (`1–180` min, `0.01–80` miles, `1–80` mph, valid NYC zones `1–265`), and exports a uniform `100,000`-row sample (`80,000` train / `20,000` test) for modeling.
2. **Basic descriptive statistics**:
   - Duration: **Mean = 16.66 min**, **Median = 12.93 min**, **Variance = 177.12**, **SD = 13.31 min**
   - Distance: **Mean = 3.60 mi**, **Median = 1.83 mi**, **Variance = 21.93**, **SD = 4.68 mi**
   - Group averages & SDs: Airport (**37.35 min**, SD 17.35) vs. Non-Airport (**14.01 min**, SD 9.98); Rush Hour (**16.93 min**, SD 14.11) vs. Non-Rush (**16.54 min**, SD 12.96)
   - Strongest correlation with duration: `trip_distance` ($r = 0.8077$) and `is_airport_trip` ($r = 0.5556$)
3. **Models compared (20,000-row untouched test split)**:

| Model | Test MAE (min) | Test RMSE (min) | Test R-squared |
|---|---:|---:|---:|
| Overall Average Baseline | 9.300 | 13.222 | -0.000 |
| If/Else Group-Average Rules | 5.969 | 9.170 | 0.519 |
| Simple Linear Regression (`8.380 + 2.300 * distance`) | 5.340 | 7.889 | 0.644 |
| Multiple Linear Regression | 4.983 | 7.422 | 0.685 |
| Tuned Ridge Regression (`alpha = 10.0`, 5-fold CV) | 4.984 | 7.422 | 0.685 |

## Where predictions are less reliable

- **Very long trips (`>10 mi`)**: Highest duration SD (`17.44 min`) and highest test MAE (`11.24 min`).
- **Airport trips**: Duration SD is `17.52 min` (test MAE `9.05 min`) due to highway, bridge/tunnel, and terminal congestion.
- **Rush-hour trips**: Higher variance (`SD = 13.70 min`, test MAE `5.29 min`) compared with non-rush trips (`SD = 13.01 min`, test MAE `4.86 min`).
