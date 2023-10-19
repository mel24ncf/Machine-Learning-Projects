import pandas as pd
import numpy as np
import sys
import util

# check command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# read in the argument
infilename = sys.argv[1]

# read the spreadsheet
X, Y, variables = util.read_excel_data(infilename)

n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")
print()

# compute coefficients for linear regression
B = np.linalg.inv(X.T @ X) @ X.T @ Y # normal equations to estimate coefficients

print(util.format_prediction(B, variables))

R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")