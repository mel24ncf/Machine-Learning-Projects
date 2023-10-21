# Supervised Learning

## Definition:
Supervised Machine Learning is the branch of machine learning that is concerned with training models on labeled datasets. These models learn from the labeled datasets to
make predictions on the labels of unseen data. In supervised learning, the goal is to predict these labels, which can be categorical, referred to as **classes**, or numeric and 
continuous, known as **targets**. The labeling determines the type of supervised task which in turn influences the set of models that one can apply to the problem. 

## Types:
### Classification:
In a classification problem, the goal is to train a computer to correctly assign each instance to its appropriate class. 
(or set of classes in multi-label classification). Examples include classifying the following:

  * Survivors of the Titanic
  * Customer Churn
  * Handwritten Digit Recognition
  * Image Classification

### Regression:
In a regression problem, the goal to train a computer to correctly predict the target for each instance. 
Examples include predictions on the following:

  * House Price 
  * College GPA 
  * Sales 

Supervised learning involves the mathematical process of finding the best parameters for a model. This optimization process is performed by minimizing a chosen loss function, which is determined by the labeled data and the parameters we seek to optimize. This function quantifies the discrepancy between what the data suggests and what our model predicts. 

To understand this concept from a geometric perspective, imagine the Loss function as a surface that extends over the feature space. In this analogy, the model parameters act as coordinates within this space. Our objective is to find the coordinate of parameters for which this surface attains a global minimum (a lowest point in the surface). 

Unfortunately, due to inability to visualize spaces beyond three dimensions, we must rely on numerical approaches for finding these parameter values. One commonly used numerical method is Gradient Descent. It is a mathematical fact that the gradient points in the direction of fastest increase, so this method operates by taking steps opposite of the direction of the gradient of the Loss function. At each step, we evaluate the gradient of the Loss function to determine which direction to move in with a certain step size or learning rate. As we approach the minimum, our steps become smaller because the gradient approaches zero, and our parameters or weights begin to converge to particular values. 
