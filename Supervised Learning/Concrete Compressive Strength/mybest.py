# Import packages
import numpy as np
import pandas as pd
from util import load_and_preprocess_data, rmse_conf_int, get_feature_names
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Create CatBoostRegressor object
cat_boost_model = CatBoostRegressor(depth=6, iterations=400, learning_rate=0.1)
pipe = Pipeline([
    ("model", cat_boost_model)
])

# Fit the model
pipe.fit(X_train, y_train)

# Compute the mean cross-validation score
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
mean_cv_score = np.mean(scores)
print(f"Mean cross-validation score: {np.abs(mean_cv_score):.2f}")

# Extract feature importances
feature_names = get_feature_names()
results = permutation_importance(pipe, X_train, y_train, scoring="neg_root_mean_squared_error")
importances = results.importances_mean

# Print feature importances
feature_importances = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
)
print(feature_importances.sort_values(by="importance", ascending=False))