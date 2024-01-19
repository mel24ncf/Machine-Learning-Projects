# Import packages
import numpy as np
from util import load_and_preprocess_data, rmse_conf_int
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Create CatBoostRegressor object
cat_boost_model = CatBoostRegressor()
pipe = Pipeline([
    ("model", cat_boost_model)
])

# Define parameters for grid search
param_grid = [
    {
        "model__learning_rate": [1, 0.5, 0.1, 0.05, 0.01],
        "model__depth": [2, 4, 6, 10],
        "model__iterations": [100, 200, 300, 400],
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

# Create CatBoostRegressor object with best hyperparameters
best_cat_boost_model = grid_search.best_estimator_

# Compute confidence interval for RMSE
y_pred = best_cat_boost_model.predict(X_test)
rmse, conf_int = rmse_conf_int(y_test, y_pred)

# Print results
print("Best hyperparameters:", best_params)
print(f"RMSE: {rmse:.2f}")
print(f"95% confidence interval: ({conf_int[0]:.2f}, {conf_int[1]:.2f})")