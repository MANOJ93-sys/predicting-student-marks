📘 Overview:

This project implements a machine learning-based regression model to predict a student's current year marks based on factors such as:

Previous year academic performance
Daily study hours
Daily sleep hours

It compares multiple machine learning models and tunes them to achieve the best performance, visualizing results through detailed plots and SHAP-based model explainability.

⚙️ Features:

✅ Synthetic dataset generation for 2000 students
✅ Exploratory Data Analysis (EDA) — statistical summaries, distributions, and correlations
✅ Data preprocessing and scaling using StandardScaler
✅ Training and evaluation of multiple models:

Linear Regression
Ridge Regression
Random Forest Regressor
Gradient Boosting Regressor
Neural Network (Keras Sequential Model)

✅ Hyperparameter tuning using GridSearchCV
✅ 10-Fold Cross Validation for performance stability
✅ Model performance visualization (MAE, R²)
✅ SHAP explainability for feature importance
✅ Interactive prediction system via user input

🧠 Technologies Used:

Python
Libraries:
pandas, numpy, matplotlib, seaborn,
scikit-learn, tensorflow, shap

📊 Performance Metrics:

Models are evaluated using:
Mean Absolute Error (MAE)
R² Score (Coefficient of Determination)

The tuned Gradient Boosting Regressor achieved the best performance, balancing accuracy and interpretability.
