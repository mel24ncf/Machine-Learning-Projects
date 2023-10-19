import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import util

# check command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# read in argument from command line
infilename = sys.argv[1]

# read spreadsheet
X, Y, variables = util.read_excel_data(infilename)

n, d = X.shape
print(f"Read {n} rows, {d-1} features from '{infilename}'.")
print()

# instantiate model
lin_reg = LinearRegression() 

# fit model
X = X[:, 1:] # ignore first column
lin_reg.fit(X, Y)

# get coefficients
intercept = lin_reg.intercept_
coefficients = lin_reg.coef_
coefficients_with_intercept = np.append(intercept, coefficients)

print(util.format_prediction(coefficients_with_intercept, variables))

# compute R2
R2 = lin_reg.score(X, Y)
print(f"R2 = {R2:f}")