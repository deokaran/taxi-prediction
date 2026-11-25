# Group 10 — Simple Taxi Trip Duration Prediction Report

## 1. Objective & Data
Predict NYC Yellow Taxi trip duration (`trip_duration_min`, minutes) using **only information available when the trip begins**.
- **Cleaned dataset**: **3,010,079** valid July 2024 trips (`100,000`-trip sample split 80/20 into 80,000 train / 20,000 test).
- **Leakage columns removed**: `tpep_dropoff_datetime`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `payment_type`, `store_and_fwd_flag`.

## 2. Basic Statistics (Average, Median, Variance, SD, Correlation)
- **Trip duration**: Mean = **16.69 min**, Median = **12.92 min**, Variance = **179.54**, SD = **13.40 min**
- **Trip distance**: Mean = **3.62 mi**, Variance = **22.01**, SD = **4.69 mi**
- **Group averages**: Airport (**37.38 min**) vs. Non-Airport (**14.05 min**); Rush Hour (**17.08 min**) vs. Non-Rush (**16.52 min**)
- **Strongest correlations**: `trip_distance` ($r = 0.8032$), `is_airport_trip` ($r = 0.5512$)

## 3. Models & Ridge Alpha Tuning
Simple Linear Regression formula: $\text{duration} = 8.395 + 2.292 \times \text{trip\_distance}$.

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

Final Tuned Ridge Regression (`alpha = 0.01`): **MAE = 5.035 min**, **RMSE = 7.635 min**, **$R^2$ = 0.681**.

## 4. Where Predictions Are Less Reliable
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

Long trips (`>10 mi`), airport trips, and rush-hour rides have the largest duration SD (highest variance), causing larger prediction errors.

## 5. Limitations & Team Contributions
- **Limitations**: Linear models do not capture sharp non-linear traffic bottlenecks; historical `trip_distance` proxies pre-trip planned route distance; weather and events are unobserved.
- **Team**: Member 1 [Name] (Cleaning) | Member 2 [Name] (Stats & EDA) | Member 3 [Name] (If/Else & Linear Regression) | Member 4 [Name] (Report & Review)
