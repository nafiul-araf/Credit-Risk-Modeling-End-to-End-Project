# Credit Risk Data Cleaning and Exploratory Data Analysis (EDA)

This repository demonstrates an end-to-end process for **data cleaning** and **exploratory data analysis (EDA)** for a credit risk dataset. The project focuses on handling data inconsistencies, outliers, and missing values, while also extracting meaningful insights through univariate and bivariate analysis. This work is intended to serve as a foundation for advanced machine learning applications, such as predicting loan defaults.

---

## Project Overview

The goal of this project is to prepare a credit risk dataset for analysis and build a strong understanding of its features. The project is divided into the following key sections:

1. **Data Cleaning**:
    - Addressing missing values.
    - Identifying and correcting inconsistent or inappropriate data.
    - Ensuring proper data types and detecting duplicate entries.
    - Handling outliers using domain knowledge and statistical techniques.

2. **Feature Engineering**:
    - Deriving new features to gain better insights.
    - Validating and filtering data based on domain-specific rules.

3. **Exploratory Data Analysis (EDA)**:
    - Univariate analysis to study the distribution of individual features.
    - Bivariate analysis to explore relationships between features and the target variable (`default`).
    - Correlation analysis to understand the strength of relationships among numerical variables.

---

## Code Walkthrough

Below is a detailed explanation of the code and its purpose:

---

### **1. Data Loading**
```python
df = pd.read_csv("/content/explored_data.csv")
df.head()
```
- The dataset is loaded into a pandas DataFrame for processing.
- The initial few rows are inspected to understand the structure and content.

---

### **2. Handling Missing Values**
```python
df.isnull().sum()
df.dropna(inplace=True)
```
- The dataset is checked for missing values using `isnull().sum()`.
- Missing values are dropped since their proportion was negligible.

---

### **3. Correcting Inconsistencies in Categorical Data**
```python
df['loan_purpose'] = df['loan_purpose'].replace({'Personaal': 'Personal'})
```
- Data inconsistencies in categorical variables (e.g., misspelled categories) are fixed using `.replace()`.

---

### **4. Data Type Corrections**
```python
df['zipcode'] = df['zipcode'].astype(str)
df['disbursal_date'] = pd.to_datetime(df['disbursal_date'])
df['installment_start_dt'] = pd.to_datetime(df['installment_start_dt'])
```
- Columns like `zipcode` are converted to strings, and date columns are converted to datetime format.

---

### **5. Checking for Duplicates**
```python
df.duplicated().sum()
```
- Duplicate records are identified, and no duplicates were found in this dataset.

---

### **6. Outlier Handling**
#### **Using the IQR Method**
```python
def iqr(column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    IQR = q3 - q1
    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)
    print(f"Lower Bound: {lower_bound} and Upper Bound: {upper_bound}\n")
```
- Interquartile Range (IQR) is used to calculate lower and upper bounds for detecting outliers.
- Each numeric column is checked iteratively.

#### **Capping Outliers**
```python
df = df[df['income'] <= df['income'].quantile(0.99)]
```
- Extreme outliers are capped using the 99th percentile threshold to retain relevant data while removing anomalies.

#### **Domain-Specific Validations**
```python
df['valid_gst'] = df['gst'] <= (df['loan_amount'] * 0.20)
df['valid_net_disbursement'] = df['net_disbursement'] <= (df['loan_amount'] - df['gst'])
df['valid_principal_outstanding'] = df['principal_outstanding'] <= df['loan_amount']
df['valid_bank_balance'] = df['bank_balance_at_application'] >= 0
```
- Additional validations are applied based on domain-specific rules (e.g., GST capped at 20% of the loan amount).

---

### **7. Univariate Analysis**
#### **Categorical Variables**
```python
cat_cols = ['gender', 'marital_status', 'employment_status', 'residence_type', 'loan_purpose', 'loan_type', 'default']
sns.countplot(data=df, x=column, ax=axes[i], palette='Set2')
```
- Bar charts are created for categorical variables to understand their distribution.

#### **Numerical Variables**
```python
sns.boxplot(data=df, y=column, ax=axes[i], palette='Set2')
```
- Boxplots are generated to visualize the spread and detect potential outliers.

---

### **8. Bivariate Analysis**
#### **Scatter Plots**
```python
sns.scatterplot(data=df, x='age', y='income', hue='default', palette='Set2')
```
- Scatterplots illustrate relationships between numerical features, such as `age` and `income`, colored by default status.

#### **Count Plots**
```python
sns.countplot(data=df, x='loan_purpose', hue='default', palette='Set2')
```
- Count plots examine the distribution of categorical variables against the target variable `default`.

#### **Heatmaps**
```python
crosstab = np.round(pd.crosstab(df['employment_status'], df['default'], normalize='index'), 2)
sns.heatmap(crosstab, annot=True, cmap='coolwarm')
```
- Cross-tabulations are visualized using heatmaps to identify patterns in categorical features.

---

### **9. Correlation Analysis**
#### **Correlation Matrix**
```python
plt.figure(figsize=(20, 20))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
```
- Correlations among numerical features are analyzed to detect multicollinearity and identify predictive relationships.

#### **Correlation with Target**
```python
target_correlation = correlation_matrix['default'].sort_values(ascending=False)
sns.barplot(x=target_correlation.index, y=target_correlation.values, palette='coolwarm')
```
- Features most correlated with the target variable (`default`) are highlighted to aid feature selection.

---

### **10. Summary**
A detailed summary is provided at the end of the analysis, highlighting key findings and ensuring data integrity for further modeling. 

#### **Outlier Handling Process**

This outlier handling process combines technical methods and domain knowledge to ensure data integrity:

1. **IQR Method**: Used to identify and address outliers in numerical columns by calculating lower and upper bounds.

2. **Quantile-Based Filtering**: Extreme values in specific columns (e.g., income) were capped using quantile thresholds to reduce the effect of outliers.

3. **Derived Features**: Additional features like `diff` (difference between loan amount and processing fee) and `pct` (percentage of processing fee relative to loan amount) were calculated for deeper insights.

4. **Domain Knowledge Validations**:
   - GST capped at 20% of the loan amount.
   - Net disbursement should not exceed the loan amount minus GST.
   - Principal outstanding should not exceed the loan amount.
   - Bank balance at the application should be non-negative.

5. **Composite Validity Check**: Combined multiple rules into a single validity flag (`valid_loan`) to ensure overall data consistency.

6. **Final Result**:
   - Number of invalid loan records: **0**

This approach ensures that both statistical anomalies and domain-specific inconsistencies are handled effectively.


#### **Univariate Analysis**

***1. Distribution of Categorical Features***

These charts provide a visual overview of the distribution of various categorical features in the loan dataset. We can see that the majority of borrowers are male and married, with most employed as salaried individuals. Owned houses are the most common residence type, and auto loans are the most frequent loan purpose. Interestingly, secured loans are significantly more popular than unsecured ones. Finally, the distribution of the target variable "default" shows that most loans were not defaulted.


***2. Distribution of Numerical Features***

This set of boxplots provides a visual representation of the distribution of various numerical features in the loan dataset. We can observe that most of the features exhibit a right-skewed distribution, indicating that a majority of the values are concentrated towards the lower end of the range. However, some features like "Loan amount" and "Processing fee" show a more uniform distribution.


#### **Bivariate Analysis**

***1. Loan Amount vs Income by Default***

This scatterplot reveals a connection between loan amount and income, with larger loans generally corresponding to higher incomes. Despite this, the majority of loans, irrespective of income level, were successfully repaid. However, a significant proportion of defaults occurred at higher income levels. This suggests that factors beyond income, such as credit history or debt-to-income ratio, could significantly influence default risk.


***2. Loan Amount vs Age by Default***

This scatterplot illustrates the relationship between loan amount and age, revealing that most borrowers are between 20 and 60 years old. While older borrowers tend to take larger loans, the majority of loans, regardless of age, were not defaulted. However, a notable portion of defaults occurred among younger borrowers. This suggests that factors beyond age, such as credit history or income level, could significantly influence default risk.


***3. Age vs Income by Default***

This scatter plot shows the relationship between age and income, with different colors representing whether a loan was defaulted or not. While there is no clear trend, it appears that defaults occur across a wide range of ages and income levels.


***4. Loan Purpose vs Default***

This bar chart shows the number of loans for each purpose (auto, home, personal, education) and whether they were defaulted or not. Overall, most loans were not defaulted. However, home loans had the highest number of defaults, while auto loans had the lowest.


***5. Loan Type vs Default***

This bar chart shows the number of secured and unsecured loans, and whether they were defaulted or not. Overall, secured loans were more common and had fewer defaults compared to unsecured loans.


***6. Proportion of Defaults by Employment Status***

This heatmap shows the proportion of defaulted and non-defaulted loans for salaried and self-employed individuals. Both salaried and self-employed individuals have a high proportion of non-defaulted loans, with only a small percentage of defaults. However, salaried individuals have a slightly lower default rate compared to self-employed individuals.


***7. Proportion of Defaults by Marital Status***

This heatmap shows the proportion of defaulted and non-defaulted loans for married and single individuals. Both married and single individuals have a high proportion of non-defaulted loans, with only a small percentage of defaults. However, married individuals have a slightly lower default rate compared to single individuals.


***8. Default Counts by City***

This bar chart shows the number of defaulted and non-defaulted loans for various cities in India. All cities have a higher number of non-defaulted loans than defaulted loans. However, Mumbai has the highest number of both defaulted and non-defaulted loans.


***9. Correlation Analysis***

This correlation matrix provides a visual representation of the relationships between various numerical features. The lighter shades indicate a stronger positive correlation, while the darker shades indicate a stronger negative correlation. We can observe several interesting relationships: There is a strong positive correlation between loan amount, sanction amount, processing fee, GST, net disbursement, and principal outstanding, suggesting that these features are closely related. Income and loan amount are also positively correlated, indicating that higher income individuals tend to take larger loans.
Age and income show a moderate positive correlation, suggesting that older individuals tend to have higher incomes. Credit utilization ratio has a moderate positive correlation with delinquent months and total DPD, suggesting that higher credit utilization is associated with a higher risk of default. Overall, the correlation matrix provides valuable insights into the relationships between different features and can be used to identify potential predictors of loan default.


***10. Correlation of Numeric Features with Default***

This analysis examines how different numerical features are related to loan defaults. The results show that a high credit utilization ratio is strongly linked to an increased chance of default, while longer loan terms have the opposite effect. Interestingly, many other factors, like loan amount or credit history length, seem to have little to no bearing on default risk. To get a clearer picture, further analysis like feature importance ranking and modelling could be helpful in understanding what truly makes a loan more likely to default.

---

## Key Takeaways
This project:
- Demonstrates the ability to clean and preprocess data effectively.
- Employs statistical and domain-specific techniques for outlier handling and validation.
- Extracts valuable insights through visualization and correlation analysis.
- Establishes a solid foundation for predictive modeling in credit risk analysis. 
