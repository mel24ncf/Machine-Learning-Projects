# Titanic: Predicting Passengers' Fate with SVM 

## 1. Summary
In this project, I use a Support Vector Machine to tackle the Kaggle challenge of predicting which passengers of the Titanic survive. \
To successfully carry out this task, I use GridSearchCV to identify the best hyperparameters for the svm classifier, with some feature engineering applied on the **Name** and **Cabin** features.  \
This model achieves an accuracy of .78708 on the Kaggle test file, placing it in the top 12% on the leaderboard.

For more information, see here https://www.kaggle.com/competitions/titanic

## 2. Features 
* PassengerId: Passenger Id
* Survived: Boolean indicating whether passenger survived **Target**
  > 0 = No \
  > 1 = Yes
* Pclass: Proxy for socio-economic status
  > 1 = Upper \
  > 2 = Middle \
  > 3 = Lower
* Name: Name of passenger
* Sex: Sex of passenger
* Age: Age of passenger in years
* SibSp: Number of siblings and spounses aboard the Titanic
* Parch: Number of parents and children aboard the Titanic
* Ticket: Ticket number
* Fare: Passenger fare
* Cabin: Cabin number
* Embarked: Port of embarkation
  > C = Cherbourg \
  > Q = Queenstown \
  > S = Southampton

## 3. Files

File              | Description
------------------|-----------------
train             | data used for training the model
titanic_eda.ipynb | Jupyter notebook for exploratory data analysis
titanic_svm.ipynb | Jupyter notebook for building the model 
