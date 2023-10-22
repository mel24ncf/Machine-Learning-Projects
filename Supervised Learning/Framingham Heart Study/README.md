# Framingham Heart Study: Logistic Regression for Coronary Heart Disease Prediction

## 1. Summary
In this project, I constructed a logistic regression model to classify patients at risk of experiencing a coronary heart disease event within the next 10 years. I also conducted a threshold analysis to identify the threshold value that maximizes the F1 score for the fitted model. In logistic regression, a threshold (typically 0.50) is the value that is used to compare against the predicted probability for classifying the positive class. 

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
* prevalentHyp: Prevalent Hypertensive, if treated or if second exam at which mean systolic was >= 140 mmHg or mean Diastolic >=90 mmHg
  > 0: Free of disease
  >
  > 1: Prevalent of disease
* Diabetes: Diabetic according to criteria of first exam treated or first exam with casual glucose >= 200 mg/dL
  > 0: Not a diabetic
  >
  > 1: Diabetic
* totChol: Serum total cholesterol mg/dL
* sysBP: Systolic blood pressure (mean of last two of three measurements) mmHg
* diaBP: Diastolic blood pressure (mean of last two of three measurements) mmHg
* BMI: Body Mass Index (weight in kilograms/ height meters squared)
* heartRate: Heart rate in bpm
* Glucose: Casual serum total cholesterol mg/dL
* TenYearCHD: Coronary heart disease event in the next 10 years
  > 0: Event did not occur
  > 
  > 1: Event occurred

## 3. Files

File            | Description
----------------|-----------------
framingham.csv  | data
chd_explore.py  | creates html report using Pandas Profiling for data summary
chd_split.py    | splits data into train (80%) and test (20%) datasets 
chd_train.py    | trains logistic regression model within data preprocessing pipeline, prints accuracy, and saves model coefficients in a pickle file.  
chd_test.py     | evaluates model on test dataset, calculates confusion matrix using a threshold of .50, performs threshold analysis
