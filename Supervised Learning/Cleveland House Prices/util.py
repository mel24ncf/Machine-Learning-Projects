import pandas as pd
import numpy as np

# read in excel file
def read_excel_data(infilename):
    '''
    Parameters:

    Input:
        infilename: string, name of excel file

    Output:
        X: observations (features) with 1s in the first column
        Y: cleveland house prices (target)
        variables: List of headers from the spreadsheet
    '''

    df = pd.read_excel(infilename, index_col=0) 
    n, d = df.values.shape 
    d = d - 1 
    X = df.values[:, :-1]
    variables = df.columns[:-1]
    Y = df.values[:, -1]
    X = np.hstack([np.ones((n, 1)), X])
    return X, Y, variables

# format predictions
def format_prediction(B, variables):
    '''
    Parameters:

    Input:
        B:         parameter estimate
        variables: list of variables for coefficients

    Output:
        pred_string: string, formatted prediction
    '''
    intercept = B[0]
    coeffs = B[1:]
    coeff_variables = variables[0:] 
    coeff_variables.insert(0, "")
    coeff_list = list(zip(coeffs, coeff_variables))
    coeff_list.sort(key=lambda x: abs(x[0]), reverse=True)
    coeffs = [x[0] for x in coeff_list]
    coeff_variables = [x[1] for x in coeff_list]
    pred_string = f"predicted house price = ${B[0]:,.2f}"
    for i in range(len(variables)):
        pred_string += f" + (${coeffs[i]:,.2f} * {coeff_variables[i]})"
    return pred_string

def score(B, X, Y):
    '''
    Parameters:

    Input:
        B: list of coefficients
        X: Observations (features)
        Y: Cleveland house prices (target)

    Output:
        R2: float, R2 score
    '''
    Y_bar = np.mean(Y)              # get mean house price
    sst_error = Y - Y_bar           # get error
    sst = sst_error @ sst_error     # get total sum of squares
    Y_hat = X @ B                   # get predicted values
    ssr_error = Y - Y_hat           # get residual error
    ssr = ssr_error @ ssr_error     # get residual sum of squares   
    R2 = 1 - (ssr/sst)
    return R2