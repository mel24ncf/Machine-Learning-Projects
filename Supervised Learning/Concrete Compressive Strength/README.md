# Concrete Compressive Strength

## 1. Summary:
This is a project from my Applied Machine Learning course, taken in the spring of 2023. The primary objective was to develop robust regression models to predict the compressive strength of concrete (csMPa) measured in Megapascals. 

Several widely used regression models were employed, including:
* Adaboost
* Catboost
* Random Forest
* Support Vector Regression

To ensure consistency and reliability across models, a dedicated utility file was implemented for data loading and preprocessing. This facilitated the creation of a standardized training and testing dataset for each model. Grid search coupled with cross-validation were used to fine-tune the hyperparameters of the models. Following the comprehensive evaluation, the Catboost model emerged as the most effective for this specific dataset. That model was retrained on the train set using its optimal hyperparameters. 

The Root Mean Squared Error (RMSE) was used to gauge the overall predictive accuracy of the selected Catboost model. The resulting mean RMSE was calculated at 4.31, providing insight into the model's precision. The permutation importance function was applied to determine the significance of each feature in the predictive model. This analysis enhances our understanding of the relative impact of different features on the model's output. To further enhance interpretability, confidence intervals for RMSE were generated. This statistical measure provides a range in which the true RMSE is likely to fall.

## 2. Data:

This data can be found on the UC Irvine Machine Learning Repository website located [here](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) \
It contains 8 features and 1 target variable. \
The data definitions below were taken directly from the UC Irvine website. \
Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable \
Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable \
Fly Ash (component 3) -- quantitative  -- kg in a m3 mixture -- Input Variable \
Water  (component 4) -- quantitative  -- kg in a m3 mixture -- Input Variable \
Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable \
Coarse Aggregate  (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable \
Fine Aggregate (component 7)	 -- quantitative  -- kg in a m3 mixture -- Input Variable \
Age -- quantitative  -- Day (1~365) -- Input Variable \
Concrete compressive strength -- quantitative -- MPa -- Output Variable

## 3. Files
This project contains six python files, and a csv file containing the data.

File                  | Description
----------------------|-----------------------
util.py               | Contains commands for loading and preprocessing the data from the csv file, as well as RMSE and feature importance functions
cs_adaboost.py        | program for the AdaBoost model
cs_catboost.py        | program for the Catboost model
cs_randomforest.py    | program for the RandomForest model
cs_supportvector.py   | program for the SupportVectorRegression model
mybest.py             | program to retrain the best model on the train data and compute the mean RMSE with feature importance
Concrete_Data_Yeh.csv | csv file containing the data

