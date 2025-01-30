# Supervised Learning

## Definition:
Supervised Machine Learning is concerned with training models to learn from labeled data. This type of learning works through a mathematical process called minimization, in which the model parameters are iteratively nudged until the error between predicted and actual labels are sufficiently small or minimized. Supervised Learning comes in two flavors: Classification (predicting groups/classes) and Regression (predicting numerical values). 

## Types:
### Classification:
In a classification problem, the goal is to train a computer to correctly assign each instance to its appropriate class 
(or set of classes in multi-label classification). Examples include:

  * Determining survivors of the Titanic
  * Predicting Customer Churn
  * Handwritten Digit Recognition
  * Image Classification

### Regression:
In a regression problem, the goal is to train a computer to correctly predict a value. 
Examples include:

  * Predicting House Price 
  * Estimating College GPA 
  * Sales Forecasting

Supervised learning involves the mathematical process of finding the best parameters for a model. This optimization process is performed by minimizing a chosen loss function, which is determined by the labeled data and the parameters we seek to optimize. This function quantifies the discrepancy between what the data suggests and what our model predicts. 

To understand this concept from a geometric perspective, imagine the Loss function as a surface that extends over the feature space. In this analogy, the model parameters act as coordinates within this space. Our objective is to find the coordinate of parameters for which this surface attains a global minimum (a lowest point). 

Unfortunately, due to our inability to visualize spaces beyond three dimensions, we must rely on numerical approaches for finding these parameter values. One commonly used algorithm is Gradient Descent. This algorithm works  by taking steps in the direction of steepest descent. At each step, we evaluate the gradient of the Loss function to determine which direction is steepest, with a certain step size or learning rate. As we approach the minimum, our steps become smaller because the gradient approaches zero, and our parameters or weights begin to converge to particular values. 
