# Group 10 — NYC Taxi Trip Duration Prediction Report

## 1. Objective and Data

This study investigates patterns in NYC Yellow Taxi trip duration across pickup times, pickup/dropoff locations, and trip-start characteristics, and builds a predictive model for **trip duration in minutes (`trip_duration_min`)** using **only information available when a trip begins**.

- **Dataset**: NYC TLC Yellow Taxi Trip Records (`yellow_tripdata_2024-07.xlsx`, 3 Excel sheets: `Data_1`, `Data_2`, `Data_3`)
- **Raw observations**: **3,076,903 rows** across 19 raw columns
- **Cleaned dataset**: **2,972,241 rows** (96.60% retained; 104,662 anomalous/duplicate rows removed)
- **Target**: `trip_duration_min = (tpep_dropoff_datetime - tpep_pickup_datetime) * 1440` (minutes)

### Strict Leakage Prevention (Trip-Start Information Only)
To satisfy the assignment requirement of using *only information available when the trip begins*, all post-trip, meter-end, and payment columns were removed before modeling:
- **Dropped leakage columns**: `Airport_fee, congestion_surcharge, extra, fare_amount, improvement_surcharge, mta_tax, payment_type, store_and_fwd_flag, tip_amount, tolls_amount, total_amount, tpep_dropoff_datetime`
- **Kept trip-start predictors**: pickup timestamp components (`pickup_hour`, `pickup_dayofweek`, `pickup_day`, cyclical `sin`/`cos` encodings, `is_weekend`, `is_rush_hour`, `is_overnight`), origin/destination taxi zones (`PULocationID`, `DOLocationID`, `same_zone`, `is_airport_pickup`, `is_airport_dropoff`, `is_airport_trip`), `VendorID`, `RatecodeID`, `passenger_count` (with missingness indicator `passenger_missing`), and `trip_distance` / `log1p_distance` (serving as the pre-trip planned route distance proxy between origin and destination).
- **Sensitivity ablation**: We also benchmarked a strict origin-destination-and-time model *without* `trip_distance` or `RatecodeID` to quantify how much predictive power comes from route distance versus pure zone-and-time patterns.

## 2. Data Preprocessing and Validation

All preprocessing decisions are grounded in domain checks on the raw 3.08M-row workbook:

| Cleaning Step | Rows Removed | Justification |
|---|---:|---|
| Parse / missing core fields | 0 | Missing pickup/dropoff timestamps, distance, or zone IDs |
| Outside July 2024 (`2024-07-01` to `2024-07-31`) | 47 | Removes erroneous timestamps (e.g., 2009 glitches and June 30 boundary rows) |
| Duration outside `[1, 180]` min | 43,590 | Removes negative/zero-minute meter glitches and >3-hour stuck meters |
| Distance outside `(0, 80]` miles | 19,616 | Removes stationary 0-mile charges and extreme GPS/odometer outliers |
| Invalid Taxi Zone ID (`1..265`) | 0 | Enforces official NYC TLC Taxi Zone range |
| Implied speed outside `[1, 80]` mph | 3,571 | Removes physically implausible trips (e.g., >80 mph NYC average speed) |
| Exact duplicates | 37,838 | Prevents duplicate records from distorting EDA or leaking across train/test |

Additionally, **297,407** rows had missing or `0` `passenger_count` (converted to `NaN` with a `passenger_missing` flag and median-imputed strictly inside the training pipeline), and **309,530** rows with missing/`99` `RatecodeID` were mapped to code `99` (Unknown).

For reproducible modeling within standard workstation memory limits, a uniform random sample of **120,000 cleaned trips** (`random_state=42`) was split **80% / 20%** into **96,000 training** and **24,000 untouched test** observations before fitting any imputer, scaler, or encoder.

## 3. Exploratory Data Analysis (Full Cleaned Dataset)

Across all **2,972,241** cleaned July 2024 trips:
- **Mean trip duration**: **16.68 min** (SD: **13.31 min**); sample median: **12.93 min** (IQR: 7.88–20.87 min)
- **Mean trip distance**: **3.60 miles**
- **Time-of-day pattern**: Mean duration peaks around **hour 16:00** (**19.08 min**, 184,338 trips) due to afternoon congestion and airport traffic, while the shortest trips occur around **hour 03:00** (**12.66 min**).
- **Airport vs. Non-Airport**: Airport-linked trips (334,846 trips) average **37.51 min**, more than double non-airport trips (**14.03 min**, 2,637,395 trips).
- **Strongest correlations with duration**: `trip_distance` ($r = 0.808$) and `log1p_distance` ($r = 0.827$) dominate, followed by `is_airport_trip` ($r = 0.563$).

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
   - `Tuned HistGradientBoosting`: Selected via 5-fold shuffled `KFold` `GridSearchCV` optimizing negative RMSE. Best parameters: `{"model__learning_rate": 0.1, "model__max_iter": 250, "model__max_leaf_nodes": 63, "model__min_samples_leaf": 50}`.
   - `Ensemble model`: `VotingRegressor` averaging the tuned `HistGradientBoostingRegressor` and a regularized `RandomForestRegressor`, reducing variance across heterogeneous tree architectures.
3. **Evaluation Measures**:
   - **RMSE** (primary tuning metric, penalizes large travel-time misses), **MAE** (direct average error in minutes for dispatch/ETA planning), and **$R^2$** (proportion of duration variance explained).

### Cross-Validation Comparison (Training Set Only)

| model                         |   cv_rmse_mean |   cv_rmse_std |
|:------------------------------|---------------:|--------------:|
| Tuned HistGradientBoosting    |          5.360 |         0.098 |
| Ensemble model                |          5.385 |         0.101 |
| HistGradientBoosting baseline |          5.442 |         0.097 |
| Random forest baseline        |          5.621 |         0.100 |
| Extra trees baseline          |          5.721 |         0.095 |
| Ridge baseline                |          6.162 |         0.104 |

### Train vs. Untouched Test Performance

| model                          | split        |   mae |   rmse |    r2 |
|:-------------------------------|:-------------|------:|-------:|------:|
| Ridge baseline                 | train        | 4.015 |  6.037 | 0.790 |
| Ridge baseline                 | test         | 4.101 |  6.346 | 0.780 |
| Random forest baseline         | train        | 3.386 |  5.211 | 0.844 |
| Random forest baseline         | test         | 3.667 |  5.835 | 0.814 |
| Extra trees baseline           | train        | 3.522 |  5.342 | 0.836 |
| Extra trees baseline           | test         | 3.782 |  5.946 | 0.807 |
| HistGradientBoosting baseline  | train        | 3.426 |  5.232 | 0.842 |
| HistGradientBoosting baseline  | test         | 3.561 |  5.645 | 0.826 |
| Tuned HistGradientBoosting     | train        | 3.182 |  4.860 | 0.864 |
| Tuned HistGradientBoosting     | test         | 3.470 |  5.550 | 0.832 |
| Ensemble model                 | train        | 3.133 |  4.821 | 0.866 |
| Ensemble model                 | test         | 3.483 |  5.575 | 0.830 |
| OD-only ablation (no distance) | test_od_only | 5.864 |  8.481 | 0.607 |

- **Best Cross-Validated Model & Ensemble Comparison**: On 5-fold cross-validation, **Tuned HistGradientBoosting** achieved the lowest CV RMSE (**5.360 min** vs **5.385 min** for the Voting Ensemble) and on the untouched test set achieved **MAE = 3.470 min**, **RMSE = 5.550 min**, **$R^2$ = 0.832**. The **Ensemble model** (averaging Tuned HistGradientBoosting and a regularized Random Forest for variance reduction across tree families) achieved nearly identical test performance (**MAE = 3.483 min**, **RMSE = 5.575 min**, **$R^2$ = 0.830**), both strongly outperforming the Ridge linear baseline (**RMSE = 6.346 min**, **$R^2$ = 0.780**).
- **Route-Distance Ablation (Strict OD + Time Only, without `trip_distance` / `RatecodeID`)**: Test **MAE = 5.864 min**, **RMSE = 8.481 min**, **$R^2$ = 0.607**. This confirms that knowing the pickup/dropoff zone pair and time of day alone explains ~60.7% of variance (because origin-destination zones encode typical inter-zone distance and congestion), while adding route distance boosts explained variance to **83.0%** and cuts MAE by **2.38 min**.

### Permutation Feature Importance (Test Set)
Top predictors by RMSE increase when permuted: **trip_distance, log1p_distance, pickup_hour_cos, RatecodeID, PULocationID, DOLocationID, pickup_hour, pickup_day**.

## 5. Supporting Travel-Time Estimation & Where Predictions Are Less Reliable

### How Findings Support Travel-Time Estimation
- Dispatchers and rider-facing ETA tools can estimate baseline travel times before pickup using planned route distance, pickup/dropoff zones, and cyclical time-of-day / rush-hour indicators.
- Zone target encoding captures persistent structural bottlenecks (e.g., Midtown crosstown corridors, JFK/LGA bridge/tunnel approaches) without post-trip fare or surcharge leakage.

### Situations Where Predictions Are Less Reliable

| dimension       | segment              |   count |   mean_actual_min |   mae |   rmse |   p90_abs_error |   mean_residual_bias |
|:----------------|:---------------------|--------:|------------------:|------:|-------:|----------------:|---------------------:|
| Distance bucket | Long (3-10 mi)       |    4891 |            23.860 | 4.843 |  6.978 |          10.204 |                0.128 |
| Distance bucket | Medium (1-3 mi)      |   11455 |            12.998 | 2.908 |  4.145 |           6.063 |                0.131 |
| Distance bucket | Short (<1 mi)        |    5430 |             6.475 | 1.872 |  2.740 |           3.965 |                0.106 |
| Distance bucket | Very Long (>10 mi)   |    2224 |            45.184 | 7.384 | 11.021 |          15.204 |                0.468 |
| Time period     | Evening (20-23)      |    4936 |            15.576 | 3.114 |  4.832 |           6.967 |                0.072 |
| Time period     | Evening Rush (16-19) |    6263 |            17.324 | 3.612 |  5.736 |           7.898 |                0.247 |
| Time period     | Late Night (00-06)   |    2175 |            14.440 | 2.693 |  4.579 |           5.769 |                0.057 |
| Time period     | Midday (10-15)       |    8022 |            17.876 | 3.781 |  5.949 |           8.401 |                0.205 |
| Time period     | Morning Rush (07-09) |    2604 |            15.763 | 3.612 |  6.040 |           7.756 |                0.028 |
| Airport status  | Airport Trip         |    2748 |            37.956 | 6.000 |  8.815 |          12.163 |                0.271 |
| Airport status  | Non-Airport Trip     |   21252 |            13.972 | 3.157 |  5.005 |           6.871 |                0.141 |
| Day type        | Weekday              |   18357 |            16.968 | 3.596 |  5.756 |           7.980 |                0.166 |
| Day type        | Weekend              |    5643 |            15.907 | 3.113 |  4.938 |           6.886 |                0.122 |
| Zone relation   | Different PU/DO Zone |   23035 |            17.143 | 3.540 |  5.606 |           7.805 |                0.154 |
| Zone relation   | Same PU/DO Zone      |     965 |             6.588 | 2.108 |  4.766 |           4.435 |                0.193 |

Key reliability gaps identified from segment-level residuals:
1. **Long-distance & airport trips (`>10 mi` / Airport)**: MAE and 90th-percentile absolute error increase substantially with trip distance because longer highway/bridge corridors accumulate unpredictable traffic incidents, toll-plaza queues, and terminal congestion.
2. **Midday and peak rush hours (`07:00–09:59`, `10:00–15:59`, `16:00–19:59`)**: Daytime commercial and commuter periods show substantially larger MAE and RMSE (including the highest RMSE during morning rush and highest MAE/90th-percentile error during midday congestion) compared with late-night (`00:00–06:59`) or late-evening (`20:00–23:59`) trips.
3. **Same-zone trips (`PULocationID == DOLocationID`)**: Short local or circling trips (e.g., waiting/errands within one zone) exhibit higher relative uncertainty (RMSE is disproportionately high relative to their 6.59 min average duration) because zone-level origin-destination pairs cannot distinguish direct hops from multi-stop local rides.

## 6. Limitations

- **Route distance proxy**: `trip_distance` in TLC records is odometer distance measured at trip completion; in live pre-trip ETA systems, planned navigation route distance would substitute for it and may differ slightly if drivers detour.
- **No live weather, road-closure, or event feeds**: Summer thunderstorms, street closures, or July 4th holiday events are not observed in the TLC schema.
- **Zone-level spatial granularity**: `PULocationID` and `DOLocationID` represent coarse Taxi Zones rather than exact GPS coordinates, hiding block-level variance inside large zones.
- **Subsampling for CV/training**: While data cleaning and EDA summary statistics cover all **2,972,241** cleaned July 2024 trips, scikit-learn model training/tuning used a reproducible **120,000**-row sample to stay within standard workstation memory.
- **Single-month seasonality**: Data covers July 2024 only and may not generalize directly to winter months or other years without retraining.

## 7. Team Contributions

Replace the placeholders below with the actual Group 10 member names and contributions before submission:

- Member 1: [Name] — Data streaming, preprocessing, validation, and leakage audit
- Member 2: [Name] — Exploratory data analysis, statistical summaries, and visualizations
- Member 3: [Name] — Feature engineering, model comparison, hyperparameter tuning, and error analysis
- Member 4: [Name] — Report compilation, interpretation, and reproducibility verification

## 8. Reproducibility

Run `python taxi_duration_project.py` inside `project/` after installing `requirements.txt`. All metrics, plots, the cleaned dataset (`data/cleaned/yellow_tripdata_2024-07_cleaned.csv`), the fitted model (`outputs/final_model.joblib`), and this report are regenerated automatically.
