# Import packages
import numpy as np
from util import load_and_preprocess_data, rmse_conf_int
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Create SVR object
svr_model = SVR()
pipe = Pipeline([
    ("model", svr_model)
])

# Define parameters for grid search
param_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": [1, 0.1, 0.01],
    "model__kernel": ["rbf", "linear"],
} 

# Create grid search object
grid_search = GridSearchCV(
    pipe,
    param_grid,
    scoring = "neg_root_mean_squared_error",
    cv = 5,
    refit = True
)

# Fit the model
grid_search.fit(X_train, y_train)

# Extract hyperparameters from best model
best_params = grid_search.best_params_

# Create SVR object with best hyperparameters
best_svr_model = grid_search.best_estimator_

# Compute confidence interval for RMSE
y_pred = best_svr_model.predict(X_test)
rmse, conf_int = rmse_conf_int(y_test, y_pred)

# Print results
print("Best hyperparameters:", best_params)
print(f"RMSE: {rmse:.2f}")
print(f"95% confidence interval: ({conf_int[0]:.2f}, {conf_int[1]:.2f})")