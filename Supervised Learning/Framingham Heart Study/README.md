# Framingham Heart Study: Logistic Regression for Coronary Heart Disease Prediction

## 1. Project Overview
This project builds a **logistic regression model** to predict whether a patient is at risk of developing **coronary heart disease (CHD) within the next 10 years**. The dataset comes from the **Framingham Heart Study**, which tracks various health indicators and lifestyle factors.

Beyond standard model evaluation, **threshold analysis** was conducted to optimize the decision threshold, improving the balance between **precision and recall** rather than relying on the default 0.50 threshold. This ensures the model is fine-tuned for real-world application in heart disease risk assessment.

## 2. Model & Methodology
- **Logistic Regression:** Chosen for interpretability and its effectiveness in binary classification tasks.
- **Data Preprocessing:** Standardization of numerical features and handling of missing values with imputation.
- **Class Imbalance Handling:** Use of `class_weight="balanced"` to address the imbalance in CHD cases.
- **Threshold Optimization:** Conducted an F1-score-based threshold analysis to improve classification performance.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and ROC-AUC were analyzed to assess model performance.

## 3. Features
| Feature Name     | Description |
|-----------------|-------------|
| **Male**        | Binary indicator for male patient (1 = Male, 0 = Female) |
| **Age**         | Age of the patient in years |
| **Education**   | Categorical variable for education level (1-4, ranging from no formal education to college degree) |
| **currentSmoker** | Binary indicator for current smoker status |
| **cigsPerDay**  | Number of cigarettes smoked per day |
| **BPMeds**      | Binary indicator for use of anti-hypertensive medication |
| **prevalentStroke** | Binary indicator for history of stroke |
| **prevalentHyp** | Binary indicator for hypertension |
| **Diabetes**    | Binary indicator for diabetes status |
| **totChol**     | Total cholesterol level (mg/dL) |
| **sysBP**       | Systolic blood pressure (mmHg) |
| **diaBP**       | Diastolic blood pressure (mmHg) |
| **BMI**         | Body Mass Index (kg/m²) |
| **heartRate**   | Heart rate (bpm) |
| **Glucose**     | Blood glucose level (mg/dL) |
| **TenYearCHD**  | Target variable: Whether CHD occurred within 10 years (1 = Yes, 0 = No) |

## 4. Files & Scripts
| File Name       | Description |
|----------------|-------------|
| **framingham.csv** | Dataset containing patient data |
| **chd_explore.py** | Generates an HTML report using Pandas Profiling for exploratory data analysis |
| **chd_split.py** | Splits the dataset into training (80%) and test (20%) sets |
| **chd_train.py** | Trains the logistic regression model, performs preprocessing, prints metrics, and saves the trained model as a pickle file |
| **chd_test.py** | Evaluates the model, calculates confusion matrix, conducts threshold analysis, and plots performance metrics (ROC curve, precision-recall tradeoff) |
| **util.py** | Holds reusable functions for plotting and saving the confusion matrix and ROC curve. |

## 5. Model Performance
### **Default Threshold (0.50):**
- **Accuracy:** 67%
- **Precision:** 25%
- **Recall:** 59%
- **F1-score:** 35%

### **Optimized Threshold (0.38):**
- **Accuracy:** 53%
- **Precision:** 22%
- **Recall:** 84%
- **F1-score:** Improved compared to the default threshold

### **Insights:**
- The optimized threshold prioritizes **higher recall**, ensuring more CHD cases are detected.
- Precision is lower, meaning more false positives occur, but this trade-off is acceptable in medical applications where missing a true CHD case is riskier than unnecessary follow-ups.
- ROC curve analysis confirms that the model provides predictive value beyond random guessing.

## 6. Future Work
- Experiment with other models (e.g., Random Forest, Gradient Boosting) for improved predictive performance.
- Implement SMOTE or other techniques to balance the dataset further.
- Incorporate additional clinical data or external risk factors to enhance predictions.

---
This project demonstrates a structured approach to **predictive modeling in healthcare** by balancing interpretability, performance, and real-world applicability. ⚕️
