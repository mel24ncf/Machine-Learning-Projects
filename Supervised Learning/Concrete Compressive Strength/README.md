# Concrete Compressive Strength

## 1. Summary:
This is a project from my Applied Machine Learning course, taken in the spring of 2023. The primary objective was to develop robust regression models to predict the compressive strength of concrete (csMPa) measure in Megapascals. 

Several widely used regression models were employed, including:
* Adaboost
* Catboost
* Random Forest
* Support Vector Regression

To ensure consistency and reliability across models, a dedicated utility file was implemented for data loading and preprocessing. This facilitated the creation of a standardized training and testing dataset for each model. Grid search coupled with cross-validation were used to fine-tune the hyperparameters of the models. Following the comprehensive evaluation, the Catboost model emerged as the most effective for this specific dataset. That model was retrained on the train set using its optimal hyperparameters. 

The mean Root Mean Squared Error (RMSE) was computed to gauge the overall predictive accuracy of the selected Catboost model. The resulting mean RMSE was calculated at 4.31, providing insight into the model's precision. The permutation importance function was applied to determine the significance of each feature in the predictive models. This analysis enhances our understanding of the relative impact of different features on the model's output. To further enhance interpretability, confidence intervals for RMSE were generated for each model. This statistical measure provides a range in which the true RMSE is likely to fall.
