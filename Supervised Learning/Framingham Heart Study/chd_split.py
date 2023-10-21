import pandas as pd 
from sklearn.model_selection import train_test_split

# read data into data frame
df = pd.read_csv("Data/Framingham/framingham.csv")

# remove rows with missing values
df = df.dropna()

# train/test split (20% to test set)
train, test = train_test_split(df, test_size=0.2)

# write out train and test sets to csv files
train.to_csv("Data/Framingham/train.csv", index=False)
test.to_csv("Data/Framingham/test.csv", index=False)