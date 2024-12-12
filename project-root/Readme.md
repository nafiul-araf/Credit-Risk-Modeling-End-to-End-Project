# **Documentation for `utils.py`**

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

---


# **Documentation for `main.py`**

## **Overview**
The `main.py` file serves as the frontend interface for a credit risk modeling system. Built with Streamlit, this application allows users to input borrower details interactively and calculate the probability of default, credit score, and risk rating. The app is designed to provide both intuitive insights and actionable outcomes, making it a practical tool for financial institutions.

---

## **Conceptual Explanation**

### **Credit Risk Modeling**
Credit risk modeling evaluates the likelihood of a borrower defaulting on a loan. The application leverages a machine learning model to assess the risk based on borrower and loan characteristics. Outputs include:
- **Default Probability**: The likelihood of loan default, expressed as a percentage.
- **Credit Score**: A numeric value (300–900) reflecting creditworthiness.
- **Credit Rating**: A qualitative rating (Poor, Average, Good, Excellent).

### **Application Features**
- **Interactive Inputs**: Users can dynamically adjust borrower and loan parameters.
- **Real-Time Risk Assessment**: Calculates default probability, credit score, and rating instantly.
- **User-Friendly Design**: Streamlit provides a clean and responsive user interface.

---

## **Detailed Code Explanation**

### 1. **Setting Up the Page**
```python
st.set_page_config(page_title="Lauki Finance: Credit Risk Modelling", page_icon="📊", layout="centered")
st.title("📊 Lauki Finance: Credit Risk Modelling")
```
- **Purpose**: Configures the app's title, icon, and layout. Creates a welcoming interface with a clear title.

### 2. **Sidebar Instructions**
```python
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Fill in the necessary fields on the right side.
    2. Adjust sliders and dropdowns for interactive inputs.
    3. Click 'Calculate Risk' to view results.
    """)
    st.image("project-root/Lauki Finance.JPG", caption="Your Trusted Finance Partner")
```
- **Purpose**: Provides clear guidance for users, ensuring ease of navigation and usage.
- **Image Integration**: Adds a logo or relevant image to enhance brand identity.

### 3. **Input Fields**
#### Customer Details
```python
col1, col2, col3 = st.columns(3)
age = col1.number_input("📅 Age", min_value=18, max_value=100, value=28, help="Enter your age (18-100).")
income = col2.number_input("💰 Income (Annual)", min_value=0, max_value=5000000, value=290875, step=50000, help="Your annual income in currency units.")
loan_amount = col3.number_input("🏦 Loan Amount", min_value=0, value=2560000, help="Total loan amount you want to borrow.")
```
- **Purpose**: Captures key demographic and financial details:
  - Age: Basic eligibility.
  - Income: Annual income in currency units.
  - Loan Amount: Requested loan amount.

#### Loan Insights
```python
lti = loan_amount / income if income > 0 else 0
st.metric(label="Loan-to-Income Ratio (LTI)", value=f"{lti:.2f}", help="This shows the ratio of the loan amount to your income.")
```
- **Purpose**: Calculates the Loan-to-Income Ratio (LTI), a critical metric for assessing affordability.

#### Loan Details
```python
loan_tenure_months = col4.slider("⏳ Loan Tenure (Months)", min_value=6, max_value=240, step=6, value=36, help="Select the loan tenure in months.")
avg_dpd_per_dm = col5.number_input("⚠ Avg DPD", min_value=0, value=0, help="Average Delinquent Days (Defaults), set to 0 if no loan history.")
dmtlm = col6.slider("📅 DMTLM (Delinquent Months to Loan Ratio)", min_value=0, max_value=100, value=0, help="Delinquency ratio, 0 if no loans.")
```
- **Purpose**: Collects loan-specific details:
  - Loan tenure in months.
  - Average delinquent days (defaults) per loan.
  - Delinquency-to-loan ratio (DMTLM).

#### Loan Purpose and Additional Details
```python
credit_utilization_ratio = col7.slider("💳 Credit Utilization (%)", min_value=0, max_value=100, value=0, help="Percentage of utilized credit, 0 if no credit.")
total_loan_months = col8.number_input("📜 Total Loan Months", min_value=0, value=0, help="Cumulative loan tenure across all loans, 0 if no loans.")
loan_purpose = col9.selectbox("🎯 Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'], help="Purpose of the loan.")
```
- **Purpose**: Captures the borrower's credit utilization, cumulative loan tenure, and loan purpose.

#### Loan and Residence Type
```python
loan_type = col10.radio("🔑 Loan Type", ['Unsecured', 'Secured'], help="Choose the type of loan.")
residence_type = col11.selectbox("🏡 Residence Type", ['Owned', 'Rented', 'Mortgage'], help="Your current residence type.")
```
- **Purpose**: Identifies the type of loan and housing situation, influencing risk assessment.

### 4. **Calculate Risk**
```python
if st.button("Calculate Risk"):
    probability, credit_score, rating = predict(...)
    st.success("✅ Risk Assessment Completed!")
    st.write(f"**Default Probability:** {probability:.2%}")
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Rating:** {rating}")
```
- **Purpose**: Triggers the prediction process when the user clicks "Calculate Risk."
- **Outputs**:
  - Default Probability: Displays as a percentage.
  - Credit Score: Numeric score (300–900).
  - Rating: Descriptive category (Poor, Average, Good, Excellent).

### 5. **Risk Insights**
```python
if rating in ['Poor', 'Average']:
    st.warning("⚠ The borrower have a high-risk credit profile. Consider improving credit habits.")
else:
    st.info("🌟 The borrower have a low-risk profile. Loan approval is likely.")
```
- **Purpose**: Provides actionable feedback based on the credit rating.

---

## **Key Features**
- **User-Centric Design**: Simplifies complex credit risk modeling for non-technical users.
- **Interactive Widgets**: Enables dynamic input adjustments and instant results.
- **Risk Insights**: Guides decision-making with clear feedback based on credit rating.

---

## **How to Use**
1. **Run the Application**:
   - Install Streamlit and required dependencies.
   - Execute `streamlit run main.py` in the terminal.
   
2. **Interact with the Interface**:
   - Enter borrower details, adjust parameters, and select loan-specific attributes.
   - Click "Calculate Risk" to view results.

3. **Integrate and Analyze**:
   - Use outputs to evaluate borrower risk profiles and make informed decisions. 


# **Documentation for Tuned Hyperparameters**

## **Overview**
The XGBoost model utilized in this project has been fine-tuned using **Optuna**, an advanced hyperparameter optimization framework. These optimized hyperparameters improve the model's performance by balancing predictive accuracy, computational efficiency, and generalization to unseen data. Below is an explanation of the selected hyperparameters and their significance.

---

## **Hyperparameter Explanation**

1. **`eta` (Learning Rate)**: `0.03962150782811734`
   - **Definition**: Controls the step size during the optimization process.
   - **Effect**: A smaller `eta` ensures more gradual learning, preventing overfitting and improving stability during training. The value of `0.0396` represents a conservative learning rate, ideal for fine-tuning.

2. **`max_depth`**: `3`
   - **Definition**: The maximum depth of each decision tree.
   - **Effect**: Limits the complexity of individual trees, preventing overfitting. A depth of `3` promotes simpler models, improving generalization to new data.

3. **`subsample`**: `0.6272358596011762`
   - **Definition**: The fraction of samples used to train each tree.
   - **Effect**: Helps prevent overfitting by using only `62.7%` of the data for training each tree. This introduces randomness and enhances the model's robustness.

4. **`colsample_bytree`**: `0.7136867658100697`
   - **Definition**: The fraction of features considered when building each tree.
   - **Effect**: Ensures diversity by using `71.4%` of the features for each tree, reducing the risk of overfitting while maintaining strong predictive power.

5. **`n_estimators`**: `388`
   - **Definition**: The number of boosting rounds or decision trees in the ensemble.
   - **Effect**: Determines the total number of trees in the model. A value of `388` provides sufficient boosting iterations to achieve high accuracy without excessive computation.

---

## **Why These Hyperparameters Matter**
The selected hyperparameters strike a balance between:
- **Performance**: Achieving high metrics such as AUC, Gini, and KS.
- **Efficiency**: Avoiding unnecessary complexity and computational overhead.
- **Generalization**: Ensuring the model performs well on new, unseen data.

---

## **Optimization Framework**
The hyperparameters were tuned using **Optuna**, which employs:
- **Bayesian Optimization**: Efficient exploration of the hyperparameter space.
- **Objective Function**: Maximizing evaluation metrics like AUC and Gini.
- **Stopping Criteria**: Automatically halts optimization when no significant improvements are observed.

---

## **Benefits of Fine-Tuning**
1. **Improved Predictive Power**: Optimal settings enhance the model’s ability to distinguish default from non-default classes.
2. **Reduced Overfitting**: Regularization through `subsample` and `colsample_bytree` ensures the model generalizes well.
3. **Efficient Training**: The chosen hyperparameters minimize unnecessary computation, making the model more deployable in production environments.

---

## **How to Apply These Hyperparameters**
If you want to replicate or adapt this model:
1. Use the following dictionary of hyperparameters in your XGBoost training function:
   ```python
   params = {
       'eta': 0.03962150782811734,
       'max_depth': 3,
       'subsample': 0.6272358596011762,
       'colsample_bytree': 0.7136867658100697,
       'n_estimators': 388
   }
   ```
2. Initialize the XGBoost classifier:
   ```python
   from xgboost import XGBClassifier
   model = XGBClassifier(**params)
   ```
3. Train the model on your dataset:
   ```python
   model.fit(X_train, y_train)
   ```
