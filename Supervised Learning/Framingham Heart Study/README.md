# Framingham Heart Study: Logistic Regression for Coronary Heart Disease Prediction

## 1. Summary
In this project, I built a classification model using logistic regression, to predict whether a person will have a coronary heart disease event within 10 years.
I also perform threshold analysis to identify the threshold that produces the largest F1 score for the fitted model. The threshold (usually .50) is a parameter that determines the positive class. 

## 2. Data
* Male: Boolean Indicator of male patient
  > 0: Female
  > 
  > 1: Male
* Age: Age of a patient in years
* Education: Categorical data for education level
  > 1: 0 to 11 years
  > 
  > 2: High School Diploma/GED
  > 
  > 3: Some College, Vocational School
  > 
  > 4: College (BS, BA) degree or more
* currentSmoker: Current cigarette smoking at exam
  > 0: Not current smoker
  > 
  > 1: Current smoker
* cigsPerDay: cigarettes smoked per day
* BPMeds: Use of Anti-hypertensive medication at exam
  > 0: Not currently used
  > 
  > 1: Current use
* prevalentStroke: Prevalent Stroke
  > 0: Free of disease
  > 
  > 1: Prevalent of disease
* prevalentHyp:
* Diabetes:
* totChol:
* sysBP:
* diaBP:
* BMI:
* heartRate:
* Glucose:
* TenYearCHD: (**Target**)

## 3. Files
This project contains four python files.

File            | Description
----------------|-----------------
framingham.csv  | data
chd_explore.py  | creates report using pandas_profiling for data exploration
data_report.html| report created from chd_explore file
chd_split.py    | splits data into train (80%) and test (20%) datasets 
chd_train.py    | fits model on pipeline containing scaler and log regression model, prints accuracy, and saves coefficients for model 
classifier.pkl  | saved scaler and weights of log regression model
chd_test.py     | model evaluation on test dataset with confusion matrix using .50 threshold, and threshold optimization 