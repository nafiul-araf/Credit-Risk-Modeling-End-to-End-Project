# Feature Engineering and Model Building

## Overview

This project demonstrates a comprehensive pipeline for feature engineering and model preparation, specifically targeting a loan default prediction task. The process includes loading data, feature engineering, handling multicollinearity, selecting the most relevant features based on Information Value (IV), encoding categorical variables, scaling numerical features, and preparing the dataset for machine learning models.

The project is structured to highlight key data preprocessing and feature selection steps, making it an excellent example of building a robust foundation for predictive modeling. Below is an explanation of the implemented steps and the Python code in detail.

---

## **1. Loading Necessary Libraries and Data**

The project starts by importing essential libraries, including:

- **Pandas** for data manipulation.
- **Numpy** for numerical operations.
- **Seaborn** and **Matplotlib** for data visualization.
- **Joblib** for saving and loading models efficiently.
- **Warnings** to suppress unnecessary warnings during execution.

Data is loaded from a CSV file (`cleaned_data.csv`), and preliminary steps include removing redundant columns (`Unnamed: 0`) and examining the dataset's shape and data types.

```python
df = pd.read_csv('/content/cleaned_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
```

---

## **2. Feature Engineering**

### 2.1 Loan-to-Income Ratio (LTI)
Loan-to-Income Ratio (LTI) is a key feature in financial analysis. It is calculated as the ratio of the loan amount to the income and rounded to two decimal places. 

```python
df['lti'] = np.round(df['loan_amount'] / df['income'], 2)
```

Visualizations:
- **KDE Plot**: To analyze the distribution of LTI by default status.
- **Histogram**: To explore LTI distribution and its relationship with default rates.

---

### 2.2 Delinquent Months to Loan Months Ratio (DMTLM)
This feature measures the percentage of months a loan has been delinquent relative to its total duration. It provides insight into the borrower's repayment behavior.

```python
df['dmtlm'] = np.round((df['delinquent_months'] / df['total_loan_months']) * 100, 1)
```

Visualizations:
- Similar KDE and histogram plots are used to study the delinquency ratio.

---

### 2.3 Average Days Past Due (DPD) per Delinquent Month
This feature calculates the average number of days past due per delinquent month, helping quantify the severity of delinquencies.

```python
df['avg_dpd_per_dm'] = np.where(
    df['delinquent_months'] > 0,
    np.round(df['total_dpd'] / df['delinquent_months'], 1),
    0
)
```

---

## **3. Dropping Irrelevant Features**

Irrelevant features such as customer IDs, location details, and dates are removed to reduce noise and avoid overfitting.

```python
df = df.drop(['cust_id', 'city', 'state', 'zipcode', 'disbursal_date', 'installment_start_dt'], axis=1)
```

---

## **4. Handling Multicollinearity**

### Variance Inflation Factor (VIF)
VIF is used to identify multicollinearity among numerical features. Features with high VIF values are iteratively removed.

```python
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data
```

After calculating VIF for both unscaled and scaled data, several features are dropped to address multicollinearity.

---

## **5. Feature Selection Using Information Value (IV)**

IV quantifies the predictive power of each feature, making it a robust metric for feature selection. Features with IV values below 0.02 are excluded.

```python
def calculate_iv_for_train(X_train, y_train):
    ...
iv_data = calculate_iv_for_train(X_train, y_train)
selected_features = iv_data[iv_data['IV'] >= 0.02]['Feature'].tolist()
```

A bar chart is plotted to visualize feature importance based on IV.

---

## **6. Encoding Categorical Variables and Scaling Numerical Features**

### Encoding
Categorical features are one-hot encoded to make them suitable for machine learning models. Missing categories in the test set are handled by aligning columns.

```python
X_train_encoded = pd.get_dummies(X_train_selected, columns=cat_cols, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
```

### Scaling
Numerical features are standardized using `StandardScaler` for better model performance.

```python
scaler = StandardScaler()
X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])
```

---

## **7. Saving Model Preparation Details**

The final processed dataset and scaling details are saved for model deployment. This ensures consistency between training and prediction pipelines.

```python
model_data = {
    'model': None,
    'scaler': scaler,
    'features': X_train_encoded.columns.tolist(),
    'cols_to_scale': num_cols,
}
```

---


## **8. Model Building and Evaluation**
The first section sets up the necessary imports and defines a helper function for training and evaluating models.

#### Key Components:
1. **Metrics Imports**:
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
   ```
   - These are used to calculate and visualize the model's performance.

2. **Helper Function**:
   ```python
   def build_evaluate_model(model, model_name, train_x, train_y, test_x, test_y):
   ```
   - A function that trains a given model and evaluates its performance.
   - **Steps within the function**:
     - Fit the model to the training data.
     - Calculate and print the training score.
     - Predict using the test data and calculate evaluation metrics:
       - **Accuracy Score**
       - **Classification Report** (precision, recall, F1-score, support)
       - **Confusion Matrix** (visualized as a heatmap).
   - The `sns.heatmap()` function from `seaborn` is used for visualizing the confusion matrix.

---

### **Baseline Models**
Three baseline models are built and evaluated using the helper function.

1. **Logistic Regression**:
   ```python
   lr_model = build_evaluate_model(
       model=LogisticRegression(), 
       model_name='Logistic Regression', 
       train_x=X_train_encoded, 
       train_y=y_train,
       test_x=X_test_encoded, 
       test_y=y_test
   )
   ```
   - A basic linear model used for binary classification tasks.

2. **Random Forest**:
   ```python
   rf_model = build_evaluate_model(
       model=RandomForestClassifier(), 
       model_name='Random Forest', 
       train_x=X_train_encoded, 
       train_y=y_train,
       test_x=X_test_encoded, 
       test_y=y_test
   )
   ```
   - An ensemble method that combines multiple decision trees.
   - It typically handles both classification and regression tasks.

3. **Extreme Gradient Boosting (XGBoost)**:
   ```python
   xgb_model = build_evaluate_model(
       model=XGBClassifier(), 
       model_name='Extreme Gradient Boost', 
       train_x=X_train_encoded, 
       train_y=y_train,
       test_x=X_test_encoded, 
       test_y=y_test
   )
   ```
   - A boosting algorithm that iteratively improves weak models by focusing on the misclassified examples from previous iterations.

---

### **Hyperparameter Tuning with Randomized Search CV**
This section tunes hyperparameters for the models using `RandomizedSearchCV`.

#### Key Components:
1. **Parameter Grids**:
   - Define the search space for each model:
     - **Logistic Regression**:
       - Vary `penalty` (regularization), `C` (regularization strength), `solver`, and `max_iter`.
     - **Random Forest**:
       - Experiment with `n_estimators` (trees), `max_depth`, `min_samples_split`, etc.
     - **XGBoost**:
       - Tune parameters like `n_estimators`, `learning_rate`, `max_depth`, etc.

2. **RandomizedSearchCV Loop**:
   ```python
   for model_name, (model, param_grid) in models.items():
   ```
   - For each model, it:
     - Initializes a `RandomizedSearchCV` object.
     - Conducts a randomized search over the defined hyperparameter space.
     - Prints and stores the best parameters.
     - Saves the best model using `joblib`.

3. **Model Evaluation**:
   After tuning, each best model is loaded and evaluated on the test data.

---

### **Hyperparameter Tuning with Optuna**
Optuna is an optimization framework used for hyperparameter tuning.

#### Key Components:
1. **Objective Functions**:
   Each function defines the search space for a specific model:
   - **Logistic Regression**:
     ```python
     def objective_logreg(trial):
         C = trial.suggest_loguniform("C", 1e-3, 1e3)
         solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
         penalty = trial.suggest_categorical("penalty", ["l1", "l2"]) if solver != "saga" else "l2"
     ```
     - Tunes regularization (`C`) and solver settings.
   - **Random Forest**:
     ```python
     def objective_rf(trial):
         n_estimators = trial.suggest_int("n_estimators", 50, 500)
         max_depth = trial.suggest_int("max_depth", 5, 30)
         min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
     ```
     - Tunes the number of trees, depth, and split criteria.
   - **XGBoost**:
     ```python
     def objective_xgb(trial):
         eta = trial.suggest_loguniform("eta", 0.01, 0.5)
         max_depth = trial.suggest_int("max_depth", 3, 10)
     ```
     - Focuses on boosting-specific parameters like learning rate (`eta`).

2. **Optimization**:
   ```python
   study_logreg.optimize(objective_logreg, n_trials=100)
   ```
   - Each model is optimized for a fixed number of trials.
   - The best parameters and corresponding accuracy scores are printed.

---

### **Evaluation of Best Models**
The best models (from `RandomizedSearchCV` and `Optuna`) are trained and evaluated on the test data.

#### Steps:
1. Train the model with the best hyperparameters.
2. Predict on the test set.
3. Calculate:
   - **Accuracy Score**
   - **Classification Report**
   - **Confusion Matrix** (visualized as a heatmap).

#### Example (Logistic Regression):
```python
best_logreg = LogisticRegression(**study_logreg.best_params, random_state=42, max_iter=1000)
best_logreg.fit(X_train_encoded, y_train)
```
- Trains the Logistic Regression model with the best hyperparameters found by Optuna.

---

---

### **Undersampling to Handle Class Imbalance**

```python
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Apply undersampling
rus = RandomUnderSampler(random_state=42)
X_train_balanced_under, y_train_balanced_under = rus.fit_resample(X_train_encoded, y_train)

# Print class distribution
print("Class distribution before undersampling:", Counter(y_train))
print("Class distribution after undersampling:", Counter(y_train_balanced_under))
```

**Explanation:**
- `RandomUnderSampler` reduces the number of majority class samples to match the minority class count.
- Balances the dataset for training and prints the class distributions before and after.

---

### **Baseline Models on Undersampled Data**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Baseline models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Evaluate models
for name, model in models.items():
    print(f"Evaluating {name} on undersampled data...")
    build_evaluate_model(model, X_train_balanced_under, y_train_balanced_under, X_test_encoded, y_test)
```

**Explanation:**
- Baseline models are trained on the undersampled dataset and evaluated using a helper function (`build_evaluate_model`).
- Prints performance metrics (accuracy, classification report, etc.).

---

### **Randomized Search CV on Undersampled Data**

```python
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter grids
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]},
    'XGBoost': {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]}
}

# Randomized search
for name, model in models.items():
    print(f"Running RandomizedSearchCV for {name}...")
    random_search = RandomizedSearchCV(model, param_grids[name], cv=3, n_iter=10, random_state=42)
    random_search.fit(X_train_balanced_under, y_train_balanced_under)
    print(f"Best params for {name}:", random_search.best_params_)
    build_evaluate_model(random_search.best_estimator_, X_train_balanced_under, y_train_balanced_under, X_test_encoded, y_test)
```

**Explanation:**
- Performs hyperparameter tuning with `RandomizedSearchCV`.
- Evaluates the best models using `build_evaluate_model`.

---

### **Optuna on Undersampled Data**

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective_logistic(trial):
    C = trial.suggest_loguniform('C', 0.01, 10)
    model = LogisticRegression(C=C, random_state=42)
    score = cross_val_score(model, X_train_balanced_under, y_train_balanced_under, cv=3).mean()
    return score

study_logistic = optuna.create_study(direction='maximize')
study_logistic.optimize(objective_logistic, n_trials=20)

print("Best Logistic Regression params:", study_logistic.best_params_)
```

**Explanation:**
- `Optuna` is used for hyperparameter tuning. An objective function defines the search space.
- The cross-validation score is optimized, and the best parameters are printed.

---

### **Oversampling to Handle Class Imbalance**

```python
from imblearn.over_sampling import SMOTE

# Apply oversampling
smote = SMOTE(random_state=42)
X_train_balanced_over, y_train_balanced_over = smote.fit_resample(X_train_encoded, y_train)

# Print class distribution
print("Class distribution before oversampling:", Counter(y_train))
print("Class distribution after oversampling:", Counter(y_train_balanced_over))
```

**Explanation:**
- `SMOTE` generates synthetic samples for the minority class, balancing the training dataset.
- Prints class distributions before and after.

---

### **Baseline Models on Oversampled Data**

```python
# Evaluate models on oversampled data
for name, model in models.items():
    print(f"Evaluating {name} on oversampled data...")
    build_evaluate_model(model, X_train_balanced_over, y_train_balanced_over, X_test_encoded, y_test)
```

**Explanation:**
- Trains and evaluates the same baseline models (Logistic Regression, Random Forest, XGBoost) on the oversampled dataset.

---

### **Randomized Search CV on Oversampled Data**

```python
# Randomized search for oversampled data
for name, model in models.items():
    print(f"Running RandomizedSearchCV for {name} on oversampled data...")
    random_search = RandomizedSearchCV(model, param_grids[name], cv=3, n_iter=10, random_state=42)
    random_search.fit(X_train_balanced_over, y_train_balanced_over)
    print(f"Best params for {name}:", random_search.best_params_)
    build_evaluate_model(random_search.best_estimator_, X_train_balanced_over, y_train_balanced_over, X_test_encoded, y_test)
```

**Explanation:**
- Similar to undersampling, `RandomizedSearchCV` tunes the models' hyperparameters for the oversampled dataset and evaluates their performance.

---

### **Optuna on Oversampled Data**

```python
def objective_xgboost(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 9)
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    score = cross_val_score(model, X_train_balanced_over, y_train_balanced_over, cv=3).mean()
    return score

study_xgboost = optuna.create_study(direction='maximize')
study_xgboost.optimize(objective_xgboost, n_trials=20)

print("Best XGBoost params:", study_xgboost.best_params_)
```

**Explanation:**
- `Optuna` is used again for fine-tuning models, but this time on the oversampled dataset.
- Returns and prints the best hyperparameters. 

---

### **Evaluation Metrics**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example evaluation
y_pred = model.predict(X_test_encoded)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

**Explanation:**
- Measures model performance with accuracy, precision, recall, and F1-score.
- Confusion matrix is visualized using a heatmap for better interpretability.

---

---

### **Code Explanation: Best Performing Models Comparison**

Below is a breakdown of each code chunk along with its explanation:

---

### **1. Load Models and Calculate Metrics**
```python
from sklearn.metrics import precision_score, recall_score

model_files = [
    "Logistic Regression_best_model_over.pkl",
    "Logistic Regression_best_model_under.pkl",
    "XGBoost_best_model_under.pkl",
    "logreg_optuna_over.pkl",
    "logreg_optuna_under.pkl",
    "lr_model_over.pkl",
    "lr_model_under.pkl",
    "xgb_model_under.pkl",
    "xgb_optuna_under.pkl"
]

results = []

for model_file in model_files:
    model = joblib.load(model_file)

    y_pred = model.predict(X_test_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)

    results.append({
        "Model": model_file,
        "Accuracy": accuracy,
        "Precision (Default)": precision,
        "Recall (Default)": recall
    })

results_df = pd.DataFrame(results)
results_df
```

**Explanation**:
- **Objective**: Load saved models and evaluate their performance on test data.
- **Process**:
  - `model_files` contains the list of models to compare.
  - For each model:
    - Load using `joblib.load`.
    - Predict labels on the test set.
    - Calculate **accuracy**, **precision**, and **recall**.
  - Append the results to a list and convert it into a DataFrame.

---

### **2. Highlight Best Models**
```python
results_df[(results_df['Precision (Default)'] == results_df['Precision (Default)'].max()) | 
           (results_df['Recall (Default)'] == results_df['Recall (Default)'].max())]
```

**Explanation**:
- **Objective**: Identify models with the highest precision or recall for the default class.
- **Logic**:
  - Filter models where precision or recall equals their respective maximum values in the dataset.

---

### **3. Visualize Model Comparison**
```python
plt.figure(figsize=(20, 8))

results_melted = results_df.melt(id_vars="Model",
                                 value_vars=["Accuracy", "Precision (Default)", "Recall (Default)"],
                                 var_name="Metric", value_name="Score")

sns.barplot(x="Score", y="Model", hue="Metric", data=results_melted, palette="Set2")

plt.title("Model Comparison Based on Test Metrics", fontsize=16)
plt.xlabel("Score", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(title="Metric", loc="upper right", fontsize=12, bbox_to_anchor=(1.4, 1))
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.show()
```

**Explanation**:
- **Objective**: Compare models visually based on accuracy, precision, and recall.
- **Process**:
  - Reshape the DataFrame to long format using `melt`.
  - Plot a grouped bar chart with `sns.barplot`, where:
    - `Model` is the y-axis.
    - `Metric` (accuracy, precision, recall) is color-coded.
    - `Score` is the x-axis.

---

### **4. Load Models for Further Evaluation**
```python
loaded_models = {model_file: joblib.load(model_file) for model_file in model_files}
```

**Explanation**:
- Load all models into a dictionary for later use in detailed evaluations.

---

### **5. Evaluate Models with Decile Statistics**
```python
from sklearn.metrics import roc_curve, roc_auc_score

def evaluate_model(model, model_name, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    df = pd.DataFrame({"default_truth": y_test, "default_probability": y_prob})
    df = df.sort_values(by="default_probability", ascending=False).reset_index(drop=True)

    df["decile"] = pd.qcut(df["default_probability"], 10, labels=range(1, 11))

    decile_stats = df.groupby("decile").agg(
        min_probability=("default_probability", "min"),
        max_probability=("default_probability", "max"),
        event_count=("default_truth", "sum"),
        non_event_count=("default_truth", lambda x: (x == 0).sum())
    )

    decile_stats["event_rate"] = decile_stats["event_count"] / decile_stats["event_count"].sum()
    decile_stats["non_event_rate"] = decile_stats["non_event_count"] / decile_stats["non_event_count"].sum()
    decile_stats["cum_event_rate"] = decile_stats["event_rate"].cumsum()
    decile_stats["cum_non_event_rate"] = decile_stats["non_event_rate"].cumsum()

    decile_stats["ks"] = abs(decile_stats["cum_event_rate"] - decile_stats["cum_non_event_rate"]) * 100
    ks_stat = decile_stats["ks"].max()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    gini_coefficient = 2 * auc_score - 1

    print(f"\n=== Model: {model_name} ===")
    print(f"KS Statistic: {ks_stat:.2f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"Gini Coefficient: {gini_coefficient:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="blue")
    plt.plot([0, 1], [0, 1], "k--", label="Random Model")
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return decile_stats
```

**Explanation**:
- **Objective**: Perform detailed evaluation of each model, focusing on:
  - Decile-based statistics.
  - ROC curve and AUC score.
  - Gini coefficient.
  - KS Statistic.
- **Process**:
  - Compute predicted probabilities.
  - Group probabilities into deciles and calculate event rates.
  - Plot ROC curve and calculate evaluation metrics.

---

### **6. Interpretability with SHAP and LIME**
```python
import shap

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_encoded)

shap.summary_plot(shap_values, X_test_encoded)
```

**Explanation**:
- **Objective**: Explain the XGBoost model using SHAP.
- **Process**:
  - Initialize SHAP explainer.
  - Generate SHAP values for test data.
  - Plot global feature importance with `summary_plot`.

```python
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=X_train_balanced_under.values,
    feature_names=X_train_balanced_under.columns,
    class_names=['Non-Default', 'Default'],
    mode='classification'
)

lime_exp = lime_explainer.explain_instance(
    data_row=X_test_encoded.iloc[0].values,
    predict_fn=xgb_model.predict_proba
)

lime_exp.show_in_notebook(show_table=True)
```

**Explanation**:
- **Objective**: Use LIME for instance-specific explanations.
- **Process**:
  - Initialize LIME with training data and features.
  - Generate explanations for a specific test instance.
  - Visualize the explanations.

---
