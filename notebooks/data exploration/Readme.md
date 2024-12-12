# Data Exploration and Familialzation

## Objectives
1. Load and explore three datasets: bureau data, loan data, and customer data.
2. Familiarize with the structure, content, and key features of each dataset.
3. Merge the datasets into a unified structure using `Customer ID` as the primary key.
4. Prepare the merged dataset for further predictive modeling and analysis.

## Dataset Description
### 1. Bureau Data
Contains details of customers' financial histories with other financial institutions. Key columns include:
- **Customer ID**: Unique identifier for each customer.
- **Credit Amount**: The total amount of credit borrowed.
- **Payment History**: Record of past payments.

### 2. Loan Data
Includes details about loans provided by the financial institution. Key columns include:
- **Customer ID**: Unique identifier for each customer.
- **Loan Amount**: The amount of the loan.
- **Loan Status**: Indicates whether the loan is active, closed, or defaulted.

### 3. Customer Data
Provides demographic and personal details of the customers. Key columns include:
- **Customer ID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Income**: Annual income of the customer.

## Process Workflow
### 1. Data Loading
The datasets were loaded using the `pandas` library to ensure flexibility in handling large data.

#### Code Snippet:
```python
import pandas as pd

# Load datasets
bureau_data = pd.read_csv('bureau.csv')
loan_data = pd.read_csv('loan.csv')
customer_data = pd.read_csv('customer.csv')
```
**Explanation**: The `read_csv` function loads CSV files into Pandas DataFrames for subsequent analysis.

### 2. Data Exploration
Exploratory Data Analysis (EDA) was conducted to understand the datasets. This included:
- Viewing the first few rows using `.head()`.
- Checking for missing values with `.isnull().sum()`.
- Analyzing the distribution of key columns with `.describe()`.

#### Code Snippet:
```python
# Display summary statistics
print(bureau_data.describe())
print(loan_data.describe())
print(customer_data.describe())

# Check for missing values
print(bureau_data.isnull().sum())
```
**Explanation**: Summary statistics and missing value counts provide insights into data quality and distribution.

### 3. Data Merging
Datasets were merged into a single DataFrame using the `merge` function on the `Customer ID` column.

#### Code Snippet:
```python
# Merge datasets
data_merged = bureau_data.merge(loan_data, on='Customer_ID', how='inner')
data_merged = data_merged.merge(customer_data, on='Customer_ID', how='inner')
```
**Explanation**: Merging ensures all relevant customer details, loan data, and bureau history are combined for analysis.

### 4. Data Cleaning
Post-merging, cleaning included:
- Handling missing values.
- Dropping irrelevant columns.
- Renaming columns for clarity.

**Explanation**: Cleaning ensures the dataset is ready for advanced analysis and modeling.

## Insights
- **Credit Patterns**: Customers with higher incomes had better credit histories.
- **Loan Status**: Default rates were higher for loans exceeding a specific amount.
- **Demographics**: Younger customers (<30 years) had higher default rates.

## Future Steps
1. Perform feature engineering to create derived metrics such as debt-to-income ratios.
2. Build predictive models (e.g., logistic regression, random forests) to classify credit risk.
3. Visualize key insights using advanced plotting libraries like `matplotlib` and `seaborn`.

## Tools and Technologies
- **Python**: Data manipulation and analysis.
- **Pandas**: Loading, merging, and cleaning data.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn** (Future): Model building and evaluation.

## Conclusion
This section lays the foundation for comprehensive credit risk analysis. It emphasizes understanding and preparing data before applying predictive techniques, ensuring accuracy and interpretability in results.

