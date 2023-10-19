# Cleveland House Prices: Linear Regression in 3 Flavors

In this project, I build a linear regression model in three different ways to model Cleveland house prices based on five features from 519 recently sold homes.

* sqft hvac: Indoor square footage
* sqft yard: Outdoor square footage
* bedrooms: Number of bedrooms
* bathrooms: Number of bathrooms
* miles to school: Distance in miles to nearest elementary school

This project contains four python files.

File         | Description
-------------|-----------------
util.py      | Contains commands for opening and handling the data from the excel file, as well as displaying the regression equation and R2 value
linreg_mi    | Solution using matrix inversion with respect to the normal equations
linreg_scikit| Solution implemented with Scikit-Learn linear regression
linreg_gd    | Solution found with gradient descent
