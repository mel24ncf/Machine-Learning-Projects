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

The main idea here is that we are trying to find the parameters that minimize a chosen Loss function that is determined by the targets we have and the parameters we are searching for. The Loss function determines a surface over the feature space,and the parameters behave as coordinates in this space. Finding the best parameters is equivalent to finding the point where this surface has a horizontal tangent plane. However, the problem is that we cannot visualize dimensions > 3, and our numerical algorithms for locating global minimums can occassionally get stuck at local minimums. 
