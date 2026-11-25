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
| `trip_duration_min` | 16.71 min | 12.95 min | 177.07 | 13.31 min |
| `trip_distance` | 3.61 mi | 1.84 mi | 21.84 | 4.67 mi |

### Group Averages & Standard Deviations
- **Airport vs. Non-Airport**: Airport trips average **37.48 min** (SD = 17.04 min) vs. **14.06 min** (SD = 10.03 min) for non-airport trips.
- **Rush Hour vs. Non-Rush**: Weekday rush-hour trips (`07–09`, `16–19`) average **17.00 min** (SD = 13.97 min) vs. **16.59 min** (SD = 13.02 min).
- **Weekday vs. Weekend**: Weekdays average **17.02 min** vs. **15.73 min** on weekends.
- **Strongest Correlation with Duration**: `trip_distance` ($r = 0.8102$) and `is_airport_trip` ($r = 0.5576$).

## 3. Modeling Approach & Hyperparameter Tuning

We progress from simple statistical baselines to linear regression:
1. **Overall Average Baseline**: Predicts the training mean (`16.71 min`) for every trip.
2. **If/Else Group-Average Rules**: Assigns the training group average using explicit `if / elif / else` rules on airport status, trip distance (`>5 mi`, `1.5–5 mi`, `<=1.5 mi`), and rush-hour flag.
3. **Simple Linear Regression**: Fits $\text{duration} = 8.358 + 2.313 \times \text{trip\_distance}$.
4. **Multiple Linear Regression**: Uses `trip_distance`, time-of-day flags (`pickup_hour`, `pickup_dayofweek`, `is_weekend`, `is_rush_hour`, `is_overnight`), `passenger_count`, `is_airport_trip`, `same_zone`, `VendorID`, and `RatecodeID` with median imputation and standardization.
5. **Tuned Ridge Linear Regression**: Tunes L2 regularization `alpha` via 5-fold `KFold` `GridSearchCV` over `[0.01, 0.1, 1.0, 10.0, 100.0]`. Selected `alpha = 10.0`.

### Ridge 5-Fold Cross-Validation Tuning

|   alpha |   cv_rmse_mean |   cv_rmse_std |
|--------:|---------------:|--------------:|
|    0.01 |         7.3556 |        0.0614 |
|    0.1  |         7.3556 |        0.0614 |
|    1    |         7.3555 |        0.0614 |
|   10    |         7.3555 |        0.0618 |
|  100    |         7.3593 |        0.0638 |

### Train & Untouched Test Results

| model                               | split   |    mae |    rmse |      r2 |
|:------------------------------------|:--------|-------:|--------:|--------:|
| Overall Average Baseline            | train   | 9.4128 | 13.319  |  0      |
| Overall Average Baseline            | test    | 9.4389 | 13.258  | -0      |
| If/Else Group-Average Rules         | train   | 5.9118 |  9.0388 |  0.5395 |
| If/Else Group-Average Rules         | test    | 5.9359 |  8.9226 |  0.5471 |
| Simple Linear Regression (Distance) | train   | 5.3208 |  7.8129 |  0.6559 |
| Simple Linear Regression (Distance) | test    | 5.331  |  7.7508 |  0.6582 |
| Multiple Linear Regression          | train   | 4.9846 |  7.3491 |  0.6955 |
| Multiple Linear Regression          | test    | 4.9846 |  7.272  |  0.6991 |
| Tuned Ridge Regression              | train   | 4.985  |  7.3492 |  0.6955 |
| Tuned Ridge Regression              | test    | 4.9852 |  7.2726 |  0.6991 |

- **Takeaway**: Predicting the overall average gives RMSE **13.258 min** ($R^2 = -0.000$). Simple `if/else` group averages cut RMSE to **8.923 min** ($R^2 = 0.547$). Linear Regression and Tuned Ridge Regression (`alpha=10.0`) improve accuracy further to **MAE = 4.985 min**, **RMSE = 7.273 min**, and **$R^2 = 0.699$**.

## 4. Travel-Time Estimation & Where Predictions Are Less Reliable

| dimension      | segment            |   trips |   mean_duration_min |   sd_duration_min |    mae |   rmse |
|:---------------|:-------------------|--------:|--------------------:|------------------:|-------:|-------:|
| Distance Group | Long (5-10 mi)     |    1779 |              26.785 |            10.958 |  7.047 |  9.976 |
| Distance Group | Medium (1.5-5 mi)  |    8166 |              16.616 |             6.872 |  4.349 |  6.099 |
| Distance Group | Short (<=1.5 mi)   |    8149 |               8.163 |             4.532 |  3.772 |  4.615 |
| Distance Group | Very Long (>10 mi) |    1906 |              44.573 |            16.095 | 10.971 | 14.548 |
| Rush Hour      | Non-Rush Hour      |   14050 |              16.678 |            13.062 |  4.879 |  7.141 |
| Rush Hour      | Rush Hour          |    5950 |              16.889 |            13.711 |  5.236 |  7.575 |
| Airport        | Airport Trip       |    2327 |              37.406 |            16.925 |  8.904 | 12.044 |
| Airport        | Non-Airport Trip   |   17673 |              14.02  |             9.879 |  4.469 |  6.384 |

- **High-variance / less reliable situations**: Longer trips (`>10 mi`, SD = 16.09 min, MAE = 10.97 min), airport trips, and peak rush hours exhibit much wider travel-time spread (higher variance/SD) due to highway congestion, toll queues, and traffic incidents.
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
