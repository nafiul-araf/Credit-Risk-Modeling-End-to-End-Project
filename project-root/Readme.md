# **GitHub Documentation for `utils.py`**

## **Overview**
This project provides a utility script to support a credit scoring system built using a machine learning model. The script includes functions to preprocess input data, make predictions, and calculate credit scores based on the probability of default. The utilities are designed to be modular, scalable, and easy to integrate into a larger credit risk evaluation pipeline.

The predictive model is trained to estimate the likelihood of loan default, and the output credit score aligns with industry standards, ranging from 300 (low creditworthiness) to 900 (excellent creditworthiness). This utility script plays a crucial role in preparing data, making predictions, and providing actionable insights.

---

## **Detailed Code Explanation**

### 1. **Model Loading**
```python
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_data = joblib.load(r"project-root/model/model_data.pkl")
```
- **Purpose**: Loads the serialized model and associated data (scaler, features, columns to scale) from a `.pkl` file.
- **Components Loaded**:
  - **`model`**: The trained machine learning model (e.g., XGBoost).
  - **`scaler`**: A `StandardScaler` object for normalizing numerical features.
  - **`features`**: The list of features used for prediction.
  - **`columns_to_scale`**: The numerical columns to be standardized.

### 2. **Data Preparation**
```python
def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    data_input = {...}
    df = pd.DataFrame([data_input])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    df = df[features]
    return df
```
- **Purpose**: Prepares user-provided input data for prediction by:
  1. Collecting raw input into a dictionary.
  2. Transforming it into a pandas DataFrame.
  3. Standardizing the specified columns using the preloaded `scaler`.
  4. Selecting only the features required by the model.

- **Key Calculations**:
  - Loan-to-income ratio (`lti`): Calculated to capture affordability. If `income` is zero, defaults to zero to avoid division errors.
  - One-hot encoding for categorical features like `loan_purpose`, `loan_type`, and `residence_type`.

### 3. **Credit Score Calculation**
```python
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    default_probability = model.predict_proba(input_df)[:, 1]
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length
    ...
    return default_probability.flatten()[0], int(credit_score), rating
```
- **Purpose**: Computes the credit score and assigns a credit rating based on model predictions.
- **Steps**:
  1. Predicts **default probability** using the model.
  2. Calculates **non-default probability** (complement of default probability).
  3. Derives the **credit score** using a linear transformation from default probability to a scale of 300–900.
  4. Determines the **credit rating**:
     - Poor: 300–499
     - Average: 500–649
     - Good: 650–749
     - Excellent: 750–900

### 4. **Prediction Function**
```python
def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
            loan_amount, loan_tenure_months, total_loan_months, 
            loan_purpose, loan_type, residence_type):
    input_df = data_preparation(...)
    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating
```
- **Purpose**: Combines data preparation and credit score calculation into a single function for streamlined predictions.
- **Inputs**:
  - User-provided data including numerical (e.g., `age`, `income`) and categorical (e.g., `loan_purpose`) features.
- **Outputs**:
  - **Probability of default**: Likelihood that the user will default on a loan.
  - **Credit score**: Numeric value representing creditworthiness.
  - **Rating**: Descriptive label (Poor, Average, Good, Excellent).

---

## **Conceptual Explanation**

### **Credit Scoring System**
The utility script is a key component of a machine learning-based credit scoring system. Credit scoring evaluates a borrower's risk of default, aiding financial institutions in decision-making for loan approvals. This implementation uses probability-based scoring to generate a credit score from 300 to 900, making it comparable with industry standards.

### **Data Transformation**
- Preprocessing ensures that the raw inputs are standardized and match the format expected by the model.
- Numerical columns are scaled using `StandardScaler`, improving model stability and performance.

### **Model Prediction**
- The trained machine learning model predicts the likelihood of default.
- The utility calculates the credit score using a base and scale length, translating default probabilities into an intuitive score range.

### **Interpretability**
- The calculated score and assigned rating provide interpretable insights into creditworthiness.
- This system ensures transparency by linking the score to measurable probabilities.

---

## **Key Features**
1. **Modular Design**: Functions are self-contained, making them reusable and adaptable.
2. **Scalability**: Can handle various input formats and extend to additional features or models.
3. **Compliance**: Credit scores align with industry norms, aiding in seamless adoption.

---

## **How to Use**

1. **Set Up the Environment**:
   - Ensure dependencies (`joblib`, `numpy`, `pandas`, `scikit-learn`) are installed.
   - Load the serialized model using `joblib.load()`.

2. **Prepare Input Data**:
   - Provide necessary inputs (age, income, loan details, etc.) to the `predict` function.

3. **Make Predictions**:
   - Call the `predict` function to obtain the probability of default, credit score, and rating.

4. **Integrate**:
   - Use the output credit score and rating for decision-making in financial workflows.

---
