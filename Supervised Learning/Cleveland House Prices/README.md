# Cleveland House Prices: Linear Regression in 3 Flavors

## 1. Summary:
In this project, I build a linear regression model in 3 different ways to predict house prices in Cleveland. \
The Multiple Linear Regression model is implemented in the following ways:
* Normal Equations
* Scikit-Learn
* Gradient Descent


## 2. Data:
* sqft hvac: Indoor square footage
* sqft yard: Outdoor square footage
* bedrooms: Number of bedrooms
* bathrooms: Number of bathrooms
* miles to school: Distance in miles to nearest elementary school
* price: price of the home (**Target**)

## 3. Files:
This project contains four python files, and an excel file containing the data.

File            | Description
----------------|-----------------
util.py         | Contains commands for opening and handling the data from the excel file, as well as displaying the regression equation and R2 value
linreg_mi.py    | Solution using matrix inversion with respect to the normal equations
linreg_scikit.py| Solution implemented with Scikit-Learn linear regression
linreg_gd.py    | Solution found with gradient descent
properties.xlsx | Dataset 
