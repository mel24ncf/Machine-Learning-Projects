import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# read train.csv into a pandas dataframe
print("Reading input...")
df_train = pd.read_csv("Data/Framingham/train.csv")

# separate df_train into Y - TenYearCHD (target) and X - all other columns (features)
X_train = df_train.drop(columns=["TenYearCHD"])
y_train = df_train["TenYearCHD"]

# create a pipeline to standardize features and fit logistic regression model
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression())])
print("Fitting...")
pipeline.fit(X_train, y_train)

# print out the accuracy of the model on the training data
print(f"Train Accuracy: {pipeline.score(X_train, y_train)}") # Train Accuracy of .853

# save the pipeline (scaling, and logistic regression coeffs) to a pickle file called classifier.pkl
pickle_path = "Machine Learning/Framingham/classifier.pkl"
print(f"Writing scaling and Logistic Regression model to {pickle_path}...")
with open(pickle_path, "wb") as pfile:
    pickle.dump(pipeline, pfile)