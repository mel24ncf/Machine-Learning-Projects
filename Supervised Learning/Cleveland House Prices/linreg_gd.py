import numpy as np
import pandas as pd
import sys
import util
import matplotlib.pyplot as plt

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

# initialize weights to zero
Bnew = np.zeros(X.shape[1])

# standardize X to improve convergence
Xp = np.copy(X)
for i in range(1, d):
    Xp[:, i] = (Xp[:, i] - np.mean(Xp[:, i])) / np.std(Xp[:, i])

# perform gradient descent to compute weights (B)
learning_rate = 0.1 
epochs = 1000           
MSEs = []

for _ in range(epochs):
    Bcurr = Bnew
    gradient = -2/n * (Xp.T @ Y - Xp.T @ Xp @ Bcurr) 
    Bnew = Bcurr - learning_rate * gradient 
    error = 1/n * np.sum((Xp @ Bnew - Y)**2)
    MSEs.append(error)
    if np.sum((Bnew - Bcurr) ** 2) < 0.01:
        break
print(f"Took {_} iterations to converge")

# Adjust weights to account for standardization
B  = np.zeros(X.shape[1])
means = X.mean(axis=0)
std = X.std(axis=0)
means[0] = -1
std[0] = 1
B[0] = Bnew @ (-1 * means / std)
B[1:] = Bnew[1:] / np.std(X[:, 1:], axis=0)

# plot error
plt.plot(MSEs)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs. Epoch")
plt.show()

# R2
R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")

# format prediction
pred_string = util.format_prediction(B, variables)
print(pred_string)