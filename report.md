# Group 10 — NYC Taxi Trip Duration Prediction Report

## 1. Project Objective & Data Source
This study investigates patterns in NYC Yellow Taxi trip duration across pickup times, airport status, and trip-start characteristics, and builds a simple, interpretable predictive model for **trip duration in minutes (`trip_duration_min`)** using **only information available when a trip begins**.

- **Source dataset**: NYC TLC Yellow Taxi Trip Records (`yellow_tripdata_2024-07.xlsx`, 3 Excel sheets)
- **Cleaned dataset**: **3,010,079** valid July 2024 trips exported to `data/cleaned/yellow_tripdata_2024-07_cleaned.csv`
- **Modeling sample**: **100,000** uniform random trips split **80% training (80,000)** and **20% untouched test (20,000)**

### Preprocessing & Leakage Prevention (Justified Choices)
- **Strict trip-start constraint**: Dropped all 12 post-trip / payment-end columns (`tpep_dropoff_datetime`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `payment_type`, `store_and_fwd_flag`) because metered fares and dropoff timestamps are only known after a trip ends.
- **Domain filtering**: Kept trips within July 2024 (`1–180` minutes, `0.01–80` miles, `1–80` mph implied speed, valid NYC Taxi Zones `1–265`) to remove negative durations, stationary 0-mile charges, GPS speed spikes, and stuck meters.
- **Missing-value handling**: Invalid/missing `passenger_count` entries were filled using the **training-set median** only (`1.0` passengers) to prevent test-set leakage.

## 2. Exploratory Data Analysis & Basic Statistics

| Metric | Trip Duration (`trip_duration_min`) | Trip Distance (`trip_distance`) |
|---|---:|---:|
| **Mean (Average)** | 16.69 min | 3.62 mi |
| **Median** | 12.92 min | 1.83 mi |
| **Variance** | 179.54 min² | 22.01 mi² |
| **Standard Deviation (SD)** | 13.40 min | 4.69 mi |

### Group Averages & Correlations
- **Airport vs. Non-Airport**: Airport-linked trips (`PULocationID` or `DOLocationID` in JFK/LGA/EWR) average **37.38 min**, more than 2.6× longer than non-airport trips (**14.05 min**).
- **Rush Hour vs. Non-Rush**: Weekday rush hours (`07:00–09:59`, `16:00–19:59`) average **17.08 min** vs. **16.52 min** outside rush hour.
- **Weekday vs. Weekend**: Weekdays average **17.00 min** vs. **15.70 min** on weekends.
- **Pearson Correlations with Duration**: `trip_distance` ($r = 0.8032$), `is_airport_trip` ($r = 0.5512$), `same_zone` ($r = -0.1532$), `passenger_count` ($r = 0.0548$), `is_overnight` ($r = -0.0468$), `is_weekend` ($r = -0.0414$).

All exploratory and diagnostic figures (`duration_distribution.png`, `duration_by_hour.png`, `duration_vs_distance.png`, `duration_by_conditions.png`, `correlation_heatmap.png`, `actual_vs_predicted.png`, `residual_distribution.png`) are saved under `outputs/figures/`.

## 3. Modeling Approach, Justification & Hyperparameter Tuning

We chose an interpretable progression from basic statistical baselines to linear regression:
1. **Overall Average Baseline**: Always predicts the training mean (`16.65 min`), establishing the zero-skill benchmark ($R^2 = 0$).
2. **If/Else Group-Average Rules**: Uses transparent `if / elif / else` branches over airport status, distance (`<1.5`, `1.5–5`, `>5` mi), and rush-hour flag with training-set averages.
3. **Simple Linear Regression**: Models $\text{duration} = 8.395 + 2.292 \times \text{trip\_distance}$ (every additional mile adds ~2.29 minutes on average).
4. **Multiple Linear Regression**: Extends Simple Linear Regression to all 9 trip-start features (`trip_distance, pickup_hour, pickup_dayofweek, is_weekend, is_rush_hour, is_overnight, passenger_count, is_airport_trip, same_zone`).
5. **Tuned Ridge Linear Regression**: Tunes L2 penalty `alpha` across `[0.01, 0.1, 1.0, 10.0, 100.0]` using 5-fold cross-validation (`neg_root_mean_squared_error`) on the training set only.

### Evaluation Metrics Justification
- **MAE (minutes)**: Directly measures average prediction error in minutes for rider/dispatcher ETAs.
- **RMSE (minutes)**: Penalizes large travel-time errors (used for cross-validation tuning).
- **$R^2$**: Measures the fraction of duration variance explained relative to predicting the overall average.

### 5-Fold Cross-Validation (`Ridge` alpha tuning)
|   alpha |   cv_rmse_mean |   cv_rmse_std |
|--------:|---------------:|--------------:|
|    0.01 |         7.6775 |        0.1242 |
|    0.1  |         7.6775 |        0.1242 |
|    1    |         7.6775 |        0.1242 |
|   10    |         7.6775 |        0.1242 |
|  100    |         7.6777 |        0.124  |

### Train & Untouched Test Results
| model                               | split   |    mae |    rmse |      r2 |
|:------------------------------------|:--------|-------:|--------:|--------:|
| Overall Average Baseline            | train   | 9.4062 | 13.3704 |  0      |
| Overall Average Baseline            | test    | 9.4829 | 13.5142 | -0.0002 |
| If/Else Group-Average Rules         | train   | 5.9253 |  9.1467 |  0.532  |
| If/Else Group-Average Rules         | test    | 5.9058 |  9.1126 |  0.5452 |
| Simple Linear Regression (Distance) | train   | 5.3582 |  7.9861 |  0.6432 |
| Simple Linear Regression (Distance) | test    | 5.335  |  7.9601 |  0.653  |
| Multiple Linear Regression          | train   | 5.0675 |  7.6631 |  0.6715 |
| Multiple Linear Regression          | test    | 5.0345 |  7.6345 |  0.6808 |
| Tuned Ridge Regression              | train   | 5.0675 |  7.6631 |  0.6715 |
| Tuned Ridge Regression              | test    | 5.0345 |  7.6345 |  0.6808 |

- **Model Progression Interpretation**: Predicting the overall mean yields Test **RMSE = 13.514 min** ($R^2 = -0.000$). Simple `if/else` group averages reduce Test RMSE to **9.113 min** ($R^2 = 0.545$). Simple Linear Regression reaches **7.960 min** ($R^2 = 0.653$), and Multiple Linear Regression / Tuned Ridge (`alpha = 0.01`) achieves **MAE = 5.035 min**, **RMSE = 7.635 min**, and **$R^2 = 0.681$**—explaining ~68.1% of trip duration variance using linear trip-start features alone, with virtually identical train and test scores (no overfitting).

### Multiple Linear Regression Coefficients (Interpretation)
| feature          |   coefficient |
|:-----------------|--------------:|
| trip_distance    |        2.6405 |
| pickup_hour      |        0.0324 |
| pickup_dayofweek |        0.1459 |
| is_weekend       |       -2.1918 |
| is_rush_hour     |        0.2216 |
| is_overnight     |       -3.1996 |
| passenger_count  |        0.1921 |
| is_airport_trip  |       -6.7295 |
| same_zone        |       -4.0631 |

- Holding other variables fixed, each additional mile adds **+2.64 min**.
- Overnight trips (`22:00–05:59`) are **3.20 min faster** (`-3.20`) and weekend trips are **2.19 min faster** (`-2.19`) due to lighter road congestion.
- Rush-hour trips add **+0.22 min**, while `is_airport_trip` shows `-6.73 min` after controlling for distance because long highway stretches to JFK/LGA/EWR move at higher average speeds per mile than dense Midtown Manhattan streets.

## 4. Supporting Travel-Time Estimation & Where Predictions Are Less Reliable

### How Findings Support Travel-Time Estimation
A simple linear formula combining planned route distance, pickup hour, weekend/overnight/rush-hour indicators, and airport flags gives dispatchers and riders an immediate, transparent baseline ETA before the trip starts—cutting error roughly in half compared with guessing the average trip duration.

### Segment Reliability Analysis (Test Set)
| dimension      | segment            |   trips |   mean_min |   sd_min |   mae |   rmse |
|:---------------|:-------------------|--------:|-----------:|---------:|------:|-------:|
| Distance Group | Long (5-10 mi)     |    1894 |      27.07 |    11.16 |  7.07 |  10.48 |
| Distance Group | Medium (1.5-5 mi)  |    8079 |      16.47 |     6.81 |  4.21 |   6    |
| Distance Group | Short (<=1.5 mi)   |    8091 |       8.11 |     4.38 |  3.83 |   4.59 |
| Distance Group | Very Long (>10 mi) |    1936 |      44.7  |    17.12 | 11.5  |  16.01 |
| Rush Hour      | Non-Rush Hour      |   14124 |      16.67 |    13.07 |  4.9  |   7.47 |
| Rush Hour      | Rush Hour          |    5876 |      17.19 |    14.51 |  5.35 |   8.02 |
| Airport        | Airport Trip       |    2325 |      37.75 |    17.14 |  9.13 |  12.64 |
| Airport        | Non-Airport Trip   |   17675 |      14.07 |    10.14 |  4.5  |   6.7  |

Predictions are **less reliable** in high-variance (high-SD) conditions:
1. **Very long trips (`>10 mi`)**: Duration SD is **17.12 min** and Test MAE rises to **11.50 min** (vs. **3.83 min** for short trips `<=1.5 mi`) because longer corridors accumulate highway delays and bottlenecks.
2. **Airport trips**: Duration SD is **17.14 min** (MAE **9.13 min**) due to bridge/tunnel congestion, toll queues, and terminal dropoff traffic.
3. **Weekday rush hours**: Higher travel-time variability (SD **14.51 min**, MAE **5.35 min**) than non-rush periods (SD **13.07 min**, MAE **4.90 min**).

## 5. Limitations
- **Linear assumption**: Linear regression cannot model sharp non-linear congestion curves or specific intersection bottlenecks.
- **Route distance proxy**: `trip_distance` in TLC logs is odometer distance recorded at trip end; in a live pre-trip system, planned navigation route distance would be used instead and may differ slightly during detours.
- **Coarse spatial & external context**: Pickup/dropoff Taxi Zones are not used individually in this linear model, and weather, road closures, or July 4th holiday events are unobserved.
- **Single-month scope**: Models are fitted on July 2024 trips only and should be retrained for winter or other seasonal periods.

## 6. Team Contributions
Replace the placeholders below with actual Group 10 member names before submission:
- Member 1: [Name] — Data cleaning, leakage removal, and validation
- Member 2: [Name] — Descriptive statistics (mean, median, variance, SD, correlation) and EDA plots
- Member 3: [Name] — If/else rule baseline, linear regression modeling, and Ridge hyperparameter tuning
- Member 4: [Name] — Reliability analysis, README instructions, and final report
