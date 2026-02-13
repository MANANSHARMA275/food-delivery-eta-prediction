# Food Delivery ETA Prediction

End-to-end machine learning project to predict **food delivery time (in minutes)** using rider, order, and geo-location features, with a deployed **Streamlit** app for real-time ETA prediction.

## 1. Problem Statement
In food delivery platforms, inaccurate ETAs reduce customer trust and increase cancellations.  
This project builds a regression-based ML system to predict delivery time more accurately than static rules.

## 2. Dataset
- Total records: **45,593**
- Target variable: `delivery_time_min`
- Input features include:
- `delivery_person_age`
- `delivery_person_ratings`
- `restaurant_latitude`, `restaurant_longitude`
- `delivery_location_latitude`, `delivery_location_longitude`
- `type_of_order`
- `type_of_vehicle`

## 3. Project Pipeline
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering (geo + categorical)
4. Model training and comparison
5. Model evaluation (RMSE / MAE / R²)
6. Deployment using Streamlit

## 4. Data Cleaning and Feature Engineering
- Standardized column names
- Removed whitespace/noisy categorical text
- Converted numeric fields safely
- Clipped unrealistic rider values (age/rating bounds)
- Fixed invalid geo rows using city-wise median strategy
- Engineered features:
- `haversine_km`
- `manhattan_proxy_km`
- `lat_diff_abs`
- `lon_diff_abs`
- `distance_bucket`
- `city`, `restaurant_code`, `rider_code` (parsed from delivery ID)

## 5. Models Trained
- Linear Regression
- Random Forest Regressor
- **CatBoost Regressor (best model)**

## 6. Model Performance
| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **CatBoost (Geo + Categorical)** | **7.2588** | **5.6974** | **0.3990** |
| Random Forest | 7.5899 | 5.9349 | 0.3430 |
| Linear Regression | 7.8313 | 6.2560 | 0.3005 |

## 7. Top Feature Importances (CatBoost)
1. `delivery_person_ratings`
2. `delivery_person_age`
3. `type_of_vehicle`
4. `manhattan_proxy_km`
5. `distance_bucket`
6. `haversine_km`

## 8. Deployment
A Streamlit app is included to perform live ETA predictions.

### App flow
Input -> Validation -> Geo Feature Calculation -> CatBoost Prediction -> ETA Output

### Validation included
- Required field checks
- Coordinate validity checks
- Auto feature engineering before inference

## 9. Project Structure
```text
.
├── Dataset.csv
├── Food_Delivery_ETA.ipynb
├── app.py
├── best_catboost_model.cbm
├── model_features.joblib
├── model_results.csv
├── top_features.csv
├── FINAL_SUBMISSION/
└── FINAL_SUBMISSION.zip
