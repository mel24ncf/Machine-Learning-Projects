import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, 
                             recall_score,
                             precision_score)
import matplotlib.pyplot as plt
import pickle

# read in test data
df_test = pd.read_csv("Data/Framingham/test.csv")

# separate df_test into Y - TenYearCHD (target) and X - all other columns (features)
X_test = df_test.drop(columns=["TenYearCHD"])
y_test = df_test["TenYearCHD"]
n, d = X_test.shape

# read in model
with open("Machine Learning/Framingham/classifier.pkl", "rb") as pfile:
    pipeline = pickle.load(pfile)

# check accuracy
test_accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy = {test_accuracy}%")

# display confusion matrix
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix with 0.5 threshold: \n{cm}")

# get un-thresholded data
y_fuzzy = pipeline.predict_proba(X_test)[:, 1]

# perform threshold analysis
threshold = 0.0
best_f1 = -1.0
thresholds = []
recall_scores = []
precision_scores = []
f1_scores = []

while threshold <= 1.0:                     # threshold is a float between 0 and 1 for classification
    thresholds.append(threshold)            # append threshold to list of thresholds
    y_pred = np.zeros(n, dtype=int)         # initialize predictions to zero
    y_pred[y_fuzzy > threshold] = 1         # set prediction to 1 if the probability is greater than the threshold
    accuracy = (y_test == y_pred).sum() / n # compare y_test to y_pred to get accuracy
    positives = y_pred.sum()                # calculate number of positive predictions
    if positives == 0:
        recall = 0.0
        precision = 1.0
    elif positives == n:
        recall = 1.0
        precision = 0.0
    else:
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
    recall_scores.append(recall)
    precision_scores.append(precision)
    if recall == 0.0 or precision == 0.0:
        f1 = 0.0
    else: 
        f1 = 2 * precision * recall / (precision + recall)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        print(f"Threshold={threshold:.3f} Accuracy={accuracy:.3f} Recall={recall:.2f} Precision={precision:.2f} F1 = {f1:.3f}")
    threshold += 0.02

y_pred = np.zeros(n, dtype=int)
y_pred[y_fuzzy > best_threshold] = 1

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix with {best_threshold} threshold: \n{cm}")

fig, ax = plt.subplots()
ax.plot(thresholds, recall_scores, label="Recall", color="blue")
ax.plot(thresholds, precision_scores, label="Precision", color="green")
ax.plot(thresholds, f1_scores, label="F1", color="red")
ax.axvline(best_threshold, 0, 1, color="red", linestyle="dashed")
ax.set_xlabel("Threshold")
ax.legend()
fig.savefig("thresholds.png")