# =============================================================================
# STUDENT PERFORMANCE PREDICTION MODEL - COMPLETE SCRIPT
# =============================================================================

# STEP 1: SETUP AND LIBRARY IMPORTS
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set a consistent style for plots
plt.style.use('seaborn-v0_8-whitegrid')
print("--- Step 1: Libraries Imported ---")

# =============================================================================
# STEP 2: SYNTHETIC DATA GENERATION
# -----------------------------------------------------------------------------
# Create a reproducible synthetic dataset of 2000 students
np.random.seed(42)
num_students =2000

# Feature 1: Previous Year Marks (out of 100)
previous_year_marks = np.clip(np.random.normal(loc=65, scale=15, size=num_students), 30, 100)

# Feature 2: Hours of Study per Day
hours_study_per_day = np.random.uniform(low=1, high=10, size=num_students)

# Feature 3: Hours of Sleep per Day
hours_sleep_per_day = np.clip(np.random.normal(loc=7, scale=1, size=num_students), 4, 9)

# Target Variable: Current Year Marks

#hours_sleep_per_day = -1 * (hours_sleep_per_day - 7)**2 + 9
# max effect = 9 when sleep = 7

noise = np.random.normal(loc=0, scale=5, size=num_students)
current_year_marks = (0.6 * previous_year_marks) + \
                     (3.5 * hours_study_per_day) + \
                     (4.0* hours_sleep_per_day) + \
                     noise
current_year_marks = np.clip(current_year_marks, 0, 100)

# Create a Pandas DataFrame
data = {
    'Previous_Year_Marks': previous_year_marks,
    'Hours_Study_Per_Day': hours_study_per_day,
    'Hours_Sleep_Per_Day': hours_sleep_per_day,
    'Current_Year_Marks': current_year_marks
}
df = pd.DataFrame(data)

print("\n--- Step 2: Data Generation Complete ---")
print("Sample of Generated Data:")
print(df.head())

# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
print("\n--- Step 3: Performing Exploratory Data Analysis ---")

# Get basic information and statistics
print("\n--- Data Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

# Visualize the distributions of each feature
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Variables', fontsize=16)
sns.histplot(df['Previous_Year_Marks'], kde=True, ax=axes[0, 0], color='skyblue').set_title('Previous Year Marks Distribution')
sns.histplot(df['Hours_Study_Per_Day'], kde=True, ax=axes[0, 1], color='salmon').set_title('Study Hours Distribution')
sns.histplot(df['Hours_Sleep_Per_Day'], kde=True, ax=axes[1, 0], color='lightgreen').set_title('Sleep Hours Distribution')
sns.histplot(df['Current_Year_Marks'], kde=True, ax=axes[1, 1], color='gold').set_title('Current Year Marks (Target) Distribution')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Visualize relationships between features using a pairplot
sns.pairplot(df, x_vars=['Previous_Year_Marks', 'Hours_Study_Per_Day', 'Hours_Sleep_Per_Day'], y_vars=['Current_Year_Marks'], height=4, aspect=1, kind='scatter', diag_kind=None)
plt.suptitle('Relationship of Features with Current Year Marks', y=1.02, fontsize=16)
plt.show()

# Visualize the correlation matrix with a heatmap
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features', fontsize=16)
plt.show()


# =============================================================================
# STEP 4: DATA PREPROCESSING AND SCALING
# -----------------------------------------------------------------------------
print("\n--- Step 4: Preprocessing and Scaling Data ---")

# 1. Define Features (X) and Target (y)
X = df[['Previous_Year_Marks', 'Hours_Study_Per_Day', 'Hours_Sleep_Per_Day']]
y = df['Current_Year_Marks']

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for clarity
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
print("\nSample of Scaled Training Data:")
print(X_train_scaled_df.head())

# =============================================================================
# STEP 5: TRAIN AND COMPARE MULTIPLE MODELS
# -----------------------------------------------------------------------------
print("\n--- Step 5: Training and Evaluating Multiple Models ---")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    #"Support Vector Regressor (SVR)": SVR(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'R2 Score': r2}
    print(f"{name}:\n  MAE: {mae:.4f}, R2 Score: {r2:.4f}")

# Convert results to a DataFrame for easy comparison
results_df = pd.DataFrame(results).T


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mae')
nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, verbose=0)

nn_pred = nn_model.predict(X_test_scaled)
print("Neural Network MAE:", mean_absolute_error(y_test, nn_pred))
print("Neural Network R2 Score:", r2_score(y_test, nn_pred))

# =============================================================================
# STEP 6: HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------
print("\n--- Step 6: Tuning the Best Model (Gradient Boosting Regressor) ---")

param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=0)
grid_search.fit(X_train_scaled, y_train)

best_gbr = grid_search.best_estimator_
print(f"Best Parameters found: {grid_search.best_params_}")

y_pred_tuned = best_gbr.predict(X_test_scaled)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

# Add tuned model results to the comparison dataframe
results_df.loc['Tuned Gradient Boosting'] = [mae_tuned, r2_tuned]
print(f"\nTuned Gradient Boosting -> MAE: {mae_tuned:.4f}, R2 Score: {r2_tuned:.4f}")

final_model = best_gbr

# =============================================================================
# CROSS-VALIDATION FOR FINAL MODEL
# -----------------------------------------------------------------------------
print("\n--- Performing Cross-Validation on the Final Model ---")

cv_results = cross_validate(final_model, X_train_scaled, y_train,
                            cv=10,
                            scoring=['r2', 'neg_mean_absolute_error'])

# Extract results
r2_scores = cv_results['test_r2']
mae_scores = -cv_results['test_neg_mean_absolute_error']

print("Cross-Validation R2 Scores:", r2_scores)
print("Mean CV R2:", r2_scores.mean())

print("\nCross-Validation MAE Scores:", mae_scores)
print("Mean CV MAE:", mae_scores.mean())

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), r2_scores, marker='o', color="blue") # Changed cv_scores to r2_scores
plt.axhline(y=r2_scores.mean(), color="red", linestyle="--", label=f"Mean R2 = {r2_scores.mean():.3f}")
plt.title("Cross-Validation R² Scores (10-Fold)", fontsize=16)
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.legend()
plt.show()

# =============================================================================
# STEP 9: ACTUAL vs PREDICTED GRAPH (Regression Line)
# -----------------------------------------------------------------------------
y_pred_final = final_model.predict(X_test_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_final, alpha=0.7, color="blue", label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2, label="Perfect Prediction")

plt.xlabel("Actual Marks", fontsize=12)
plt.ylabel("Predicted Marks", fontsize=12)
plt.title("Actual vs Predicted Marks (Final Model)", fontsize=16)
plt.legend()
plt.show()



# =============================================================================
# STEP 7: FINAL MODEL COMPARISON AND VISUALIZATION
# -----------------------------------------------------------------------------
print("\n--- Step 7: Visualizing Final Model Comparison ---")

results_df_sorted = results_df.sort_values('R2 Score', ascending=False)
print("\n--- Model Performance Summary ---")
print(results_df_sorted)

fig, ax1 = plt.subplots(figsize=(12, 7))
sns.barplot(x=results_df_sorted.index, y='R2 Score', data=results_df_sorted, ax=ax1, palette='viridis')
ax1.set_ylabel('R-squared (R2) Score', fontsize=14)
ax1.set_xlabel('Model', fontsize=14)
ax1.set_title('Model Performance Comparison', fontsize=18)
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
sns.lineplot(x=results_df_sorted.index, y='MAE', data=results_df_sorted, ax=ax2, color='red', marker='o', label='MAE')
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.legend(loc='upper right')
plt.show()

import shap

explainer = shap.Explainer(final_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Waterfall plot for one student
shap.plots.waterfall(shap_values[0])

# Summary plot for overall importance
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

# =============================================================================
# STEP 8: MAKE A PREDICTION WITH THE FINAL MODEL (DYNAMIC WITH WHILE LOOP)
# -----------------------------------------------------------------------------
print("\n--- Step 8: Making Predictions Dynamically ---")

final_model = best_gbr

while True:
    try:
        # Take user inputs
        pmarks = float(input("Enter Previous Year Marks (0-100): "))
        study = float(input("Enter Study Hours per Day (0-10): "))
        sleep = float(input("Enter Sleep Hours per Day (3-10): "))

        # Validation checks
        if pmarks < 0 or pmarks > 100:
            print("❌ ERROR: Previous Year Marks must be between 0 and 100.\n")
            continue
        if sleep < 3 or sleep > 10:
            print("❌ ERROR: Sleep hours must be between 3 and 10.\n")
            continue
        if study < 0 or study > 10:
            print("❌ ERROR: Study hours must be between 0 and 10.\n")
            continue

        # Prepare data for prediction
        new_student_data = np.array([[pmarks, study, sleep]])
        new_student_data_scaled = scaler.transform(new_student_data)
        predicted_marks = final_model.predict(new_student_data_scaled)

        # Show results
        print("\nHypothetical Student Profile:")
        print(f"  - Previous Year Marks: {pmarks}")
        print(f"  - Hours Study Per Day: {study}")
        print(f"  - Hours Sleep Per Day: {sleep}")
        print(f"\n🎯 Predicted Current Year Marks: {predicted_marks[0]:.2f}\n")

    except ValueError:
        print("❌ ERROR: Please enter numeric values only.\n")
        continue

    # Exit condition
    choice = input("Do you want to predict another student? (yes/no): ").strip().lower()
    if choice not in ['yes', 'y']:
        print("\n--- Prediction Session Ended ---")
        break