import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Base models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

# ========================
# LOAD DATASET
# ========================
csv_file_path = "synthetic_student_performance.csv"
df = pd.read_csv(csv_file_path)
print(f"Loaded dataset with {len(df)} samples.")

# ========================
# PREPARE DATA & FEATURES
# ========================
# We use the features that were used in the Final Grade formula:
# Final_Grade = 0.4*Previous_Grades + 0.2*Study_Hours_Per_Week + 0.15*Attendance_Percentage
#               - 0.1*Number_of_Absences + 0.1*Engagement_Score - 0.05*(Social_Media_Usage*5)
features = [
    "Previous_Grades",
    "Study_Hours_Per_Week",
    "Attendance_Percentage",
    "Number_of_Absences",
    "Engagement_Score",
    "Social_Media_Usage"
]
target = "Final_Grade"

X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data (10% test set)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# ========================
# DEFINE BASE MODELS FOR STACKING
# ========================
# XGBoost Model
xgb_model = XGBRegressor(
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

# LightGBM Model
lgb_model = LGBMRegressor(
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

# Random Forest Model
rf_model = RandomForestRegressor(
    random_state=42,
    n_estimators=300,
    max_depth=10
)

# Define a list of estimators for stacking
estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('rf', rf_model)
]

# Final estimator for stacking
final_estimator = LinearRegression()

# Create the stacking regressor
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    n_jobs=-1
)

# ========================
# TRAIN THE STACKING MODEL
# ========================
stacking_reg.fit(X_train, y_train)

# Evaluate the stacking model on training and test sets
y_train_pred_stack = stacking_reg.predict(X_train)
y_test_pred_stack = stacking_reg.predict(X_test)

print("\nStacking Ensemble Model Performance:")
print(f"Training R² Score: {r2_score(y_train, y_train_pred_stack)*100:.2f}%")
print(f"Testing R² Score: {r2_score(y_test, y_test_pred_stack)*100:.2f}%")
print(f"Training MAE: {mean_absolute_error(y_train, y_train_pred_stack):.2f}")
print(f"Testing MAE: {mean_absolute_error(y_test, y_test_pred_stack):.2f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred_stack)):.2f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_stack)):.2f}")

# ========================
# USER INPUT, PREDICTION & HYPOTHESIS CHECK
# ========================
print("\nEnter Student Details for Testing:")

# Gather user input for each feature
user_input = {}
user_input["Previous_Grades"] = float(input("Previous Grades (out of 100): "))
user_input["Study_Hours_Per_Week"] = float(input("Study Hours Per Week: "))
user_input["Attendance_Percentage"] = float(input("Attendance Percentage: "))
user_input["Number_of_Absences"] = float(input("Number of Absences: "))
user_input["Engagement_Score"] = float(input("Engagement Score (1-10): "))
user_input["Social_Media_Usage"] = float(input("Social Media Usage (hours per day): "))

# Convert user input to DataFrame and standardize
user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)

# Model prediction using the stacking ensemble
predicted_grade_stack = stacking_reg.predict(user_df_scaled)[0]

# Compute actual final grade using the original formula:
# Final_Grade = 0.4*Previous_Grades + 0.2*Study_Hours_Per_Week + 0.15*Attendance_Percentage
#               - 0.1*Number_of_Absences + 0.1*Engagement_Score - 0.05*(Social_Media_Usage*5)
actual_grade = (
    0.4 * user_input["Previous_Grades"] +
    0.2 * user_input["Study_Hours_Per_Week"] +
    0.15 * user_input["Attendance_Percentage"] -
    0.1 * user_input["Number_of_Absences"] +
    0.1 * user_input["Engagement_Score"] -
    0.05 * user_input["Social_Media_Usage"] * 5
)
# Clip actual grade between 50 and 100 (as per our dataset creation)
actual_grade = int(np.clip(actual_grade, 50, 100))

print(f"\nActual Final Grade (computed using formula): {actual_grade}")
print(f"Predicted Final Grade using Stacking Ensemble: {predicted_grade_stack:.2f}")

# Hypothesis Check: Define a threshold (e.g., ±5 points)
threshold = 5
error_stack = abs(actual_grade - predicted_grade_stack)
print(f"Absolute error: {error_stack:.2f}")

if error_stack <= threshold:
    print("Hypothesis Check: PASS - The stacking ensemble prediction is within the acceptable error threshold.")
else:
    print("Hypothesis Check: FAIL - The stacking ensemble prediction is not within the acceptable error threshold.")

# ========================
# VISUALIZATION: Compare actual vs. predicted on test set (Stacking Ensemble)
# ========================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred_stack, color='blue', alpha=0.6, label="Stacking Ensemble Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Fit")
plt.xlabel("Actual Final Grades")
plt.ylabel("Predicted Final Grades")
plt.title("Stacking Ensemble: Actual vs. Predicted Final Grades")
plt.legend()
plt.show()

import pickle

# After training your stacking ensemble and creating 'scaler'...
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("stacking_model.pkl", "wb") as f:
    pickle.dump(stacking_reg, f)
