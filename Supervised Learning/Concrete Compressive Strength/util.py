# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import t

# Define function to load and preprocess data
def load_and_preprocess_data(filepath = "../../Data/Concrete/Concrete_Data_Yeh.csv"):
    """
    Load and preprocess data.
    
    Parameters
    ----------
    filepath : str
        Filepath to the data.
    
    Returns
    -------
    X_train : numpy.ndarray
        Training data.
    y_train : pandas.Series
        Training labels.
    X_test : numpy.ndarray
        Test data.
    y_test : pandas.Series
        Test labels.
    """
    # Load data
    df = pd.read_csv(filepath)
    df = df.dropna() # Drop rows with missing values
    
    # Split data into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train, X_test, y_test = (
        train.drop("csMPa", axis=1),
        train["csMPa"],
        test.drop("csMPa", axis=1),
        test["csMPa"]
    )
    
    # Create pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])
    
    X_train = pipe.fit_transform(X_train) # Fit and transform training data
    X_test = pipe.transform(X_test) # Transform test data
    
    return X_train, y_train, X_test, y_test

# Define function to compute confidence interval for RMSE
def rmse_conf_int(y_true, y_pred, alpha=0.05):
    """
    Compute confidence interval for RMSE.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    alpha : float
        Significance level.
    
    Returns
    -------
    conf_int : tuple
        Confidence interval.
    """
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate 95% confidence interval
    n = len(y_true)
    dof = n - 1
    t_val = t.ppf(1 - alpha / 2, dof)
    moe = t_val * rmse / np.sqrt(n)
    conf_int = (rmse - moe, rmse + moe)
    
    return rmse, conf_int

# Define function to provide feature names
def get_feature_names(filepath = "../../Data/Concrete/Concrete_Data_Yeh.csv"):
    """
    Return feature names.
    
    Returns
    -------
    feature_names : list
        List of feature names.
    """
    df = pd.read_csv(filepath)
    feature_names = df.drop("csMPa", axis=1).columns.tolist()

    return feature_names