# Import packages
import numpy as np
from util import load_and_preprocess_data, rmse_conf_int
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Create RandomForestRegressor object
random_forest_model = RandomForestRegressor()
pipe = Pipeline([
    ("model", random_forest_model)
])

# Define parameters for grid search
param_grid = [
    {
        "model__n_estimators": [30, 50, 100, 150, 200, 250],
        "model__max_depth": [2, 3, 4],
        "model__max_features": [2, 3, 4, 8],
        "model__min_samples_split": [2, 3, 4],
        "model__random_state": [42],
    }
]

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

# Create RandomForestRegressor object with best hyperparameters
best_random_forest_model = grid_search.best_estimator_

# Compute confidence interval for RMSE
y_pred = best_random_forest_model.predict(X_test)
rmse, conf_int = rmse_conf_int(y_test, y_pred)

# Print results
print("Best hyperparameters:", best_params)
print(f"RMSE: {rmse:.2f}")
print(f"95% confidence interval: ({conf_int[0]:.2f}, {conf_int[1]:.2f})")