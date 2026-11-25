# Group 10 — Simple Taxi Trip Duration Prediction Report

## 1. Objective and Data
This project investigates patterns in NYC Yellow Taxi trip duration using basic statistical analysis and builds a simple, interpretable predictive model for **trip duration in minutes (`trip_duration_min`)** using **only information available when a trip begins**.

- **Raw dataset**: `yellow_tripdata_2024-07.xlsx` (3 sheets)
- **Cleaned rows**: **3,010,079** valid July 2024 trips (`1–180` min, `0.01–80` miles, `1–80` mph, valid NYC zone IDs `1–265`)
- **Analysis sample**: **100,000** uniform random trips split **80% train (80,000) / 20% test (20,000)**
- **Strict leakage removal**: Dropped all 12 post-trip columns (`tpep_dropoff_datetime, fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount, congestion_surcharge, Airport_fee, payment_type, store_and_fwd_flag`)

## 2. Basic Statistical Summary (Average, Median, Variance, SD, Correlation)

| Variable | Mean (Average) | Median | Variance | Standard Deviation (SD) |
|---|---:|---:|---:|---:|
| `trip_duration_min` | 16.66 min | 12.93 min | 177.12 | 13.31 min |
| `trip_distance` | 3.60 mi | 1.83 mi | 21.93 | 4.68 mi |

### Group Averages & Standard Deviations
- **Airport vs. Non-Airport**: Airport trips average **37.35 min** (SD = 17.35 min) vs. **14.01 min** (SD = 9.98 min) for non-airport trips.
- **Rush Hour vs. Non-Rush**: Weekday rush-hour trips (`07–09`, `16–19`) average **16.93 min** (SD = 14.11 min) vs. **16.54 min** (SD = 12.96 min).
- **Weekday vs. Weekend**: Weekdays average **16.93 min** vs. **15.79 min** on weekends.
- **Strongest Correlation with Duration**: `trip_distance` ($r = 0.8077$) and `is_airport_trip` ($r = 0.5556$).

## 3. Modeling Approach & Hyperparameter Tuning

We progress from simple statistical baselines to linear regression:
1. **Overall Average Baseline**: Predicts the training mean (`16.66 min`) for every trip.
2. **If/Else Group-Average Rules**: Assigns the training group average using explicit `if / elif / else` rules on airport status, trip distance (`>5 mi`, `1.5–5 mi`, `<=1.5 mi`), and rush-hour flag.
3. **Simple Linear Regression**: Fits $\text{duration} = 8.380 + 2.300 \times \text{trip\_distance}$.
4. **Multiple Linear Regression**: Uses `trip_distance`, time-of-day flags (`pickup_hour`, `pickup_dayofweek`, `is_weekend`, `is_rush_hour`, `is_overnight`), `passenger_count`, `is_airport_trip`, `same_zone`, `VendorID`, and `RatecodeID` with median imputation and standardization.
5. **Tuned Ridge Linear Regression**: Tunes L2 regularization `alpha` via 5-fold `KFold` `GridSearchCV` over `[0.01, 0.1, 1.0, 10.0, 100.0]`. Selected `alpha = 10.0`.

### Ridge 5-Fold Cross-Validation Tuning

|   alpha |   cv_rmse_mean |   cv_rmse_std |
|--------:|---------------:|--------------:|
|    0.01 |         7.3733 |        0.0802 |
|    0.1  |         7.3733 |        0.0802 |
|    1    |         7.3733 |        0.0803 |
|   10    |         7.3733 |        0.0809 |
|  100    |         7.3784 |        0.0843 |

### Train & Untouched Test Results

| model                               | split   |    mae |    rmse |      r2 |
|:------------------------------------|:--------|-------:|--------:|--------:|
| Overall Average Baseline            | train   | 9.4016 | 13.3303 |  0      |
| Overall Average Baseline            | test    | 9.2998 | 13.2221 | -0      |
| If/Else Group-Average Rules         | train   | 5.8947 |  9.0467 |  0.5394 |
| If/Else Group-Average Rules         | test    | 5.9693 |  9.1696 |  0.519  |
| Simple Linear Regression (Distance) | train   | 5.3121 |  7.8328 |  0.6547 |
| Simple Linear Regression (Distance) | test    | 5.3397 |  7.8889 |  0.644  |
| Multiple Linear Regression          | train   | 4.9763 |  7.3645 |  0.6948 |
| Multiple Linear Regression          | test    | 4.9833 |  7.4224 |  0.6849 |
| Tuned Ridge Regression              | train   | 4.9768 |  7.3646 |  0.6948 |
| Tuned Ridge Regression              | test    | 4.9839 |  7.4221 |  0.6849 |

- **Takeaway**: Predicting the overall average gives RMSE **13.222 min** ($R^2 = -0.000$). Simple `if/else` group averages cut RMSE to **9.170 min** ($R^2 = 0.519$). Linear Regression and Tuned Ridge Regression (`alpha=10.0`) improve accuracy further to **MAE = 4.984 min**, **RMSE = 7.422 min**, and **$R^2 = 0.685$**.

## 4. Travel-Time Estimation & Where Predictions Are Less Reliable

| dimension      | segment            |   trips |   mean_duration_min |   sd_duration_min |    mae |   rmse |
|:---------------|:-------------------|--------:|--------------------:|------------------:|-------:|-------:|
| Distance Group | Long (5-10 mi)     |    1895 |              26.919 |            11.317 |  7.071 | 10.496 |
| Distance Group | Medium (1.5-5 mi)  |    8229 |              16.466 |             6.993 |  4.297 |  6.153 |
| Distance Group | Short (<=1.5 mi)   |    8063 |               8.138 |             4.449 |  3.788 |  4.627 |
| Distance Group | Very Long (>10 mi) |    1813 |              44.002 |            17.441 | 11.242 | 15.017 |
| Rush Hour      | Non-Rush Hour      |   14032 |              16.478 |            13.012 |  4.855 |  7.311 |
| Rush Hour      | Rush Hour          |    5968 |              16.87  |            13.7   |  5.287 |  7.678 |
| Airport        | Airport Trip       |    2297 |              36.403 |            17.515 |  9.048 | 12.403 |
| Airport        | Non-Airport Trip   |   17703 |              14.025 |            10.01  |  4.457 |  6.502 |

- **High-variance / less reliable situations**: Longer trips (`>10 mi`, SD = 17.44 min, MAE = 11.24 min), airport trips, and peak rush hours exhibit much wider travel-time spread (higher variance/SD) due to highway congestion, toll queues, and traffic incidents.
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
