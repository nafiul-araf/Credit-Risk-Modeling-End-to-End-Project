# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:15:47 2024

@author: Admin
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


model_data = joblib.load(r"D:\Data Science and Data Analytics\ML\Assignment - Credit Risk Modelling\Credit Risk Modelling\Project Files\model\model_data.pkl")
model_data


# =============================================================================
# import importlib
# 
# # List of libraries to check
# libraries = [
#     "joblib",
#     "pandas",
#     "numpy",
#     "streamlit",
#     "sklearn",
#     "xgboost"
# ]
# 
# # Loop through each library and print its version
# for library in libraries:
#     try:
#         module = importlib.import_module(library)
#         print(f"{library} version: {module.__version__}")
#     except ImportError:
#         print(f"{library} is not installed.")
#     except AttributeError:
#         print(f"Unable to determine version for {library}.")
# =============================================================================

model = model_data['model']
print(model)

scaler = model_data['scaler']
print(scaler)

features = model_data['features']
print(features)
len(features)

columns_to_scale = model_data['cols_to_scale']
print(columns_to_scale)


def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    data_input = {'age': age,
                  'avg_dpd_per_dm': avg_dpd_per_dm,
                  'credit_utilization_ratio': credit_utilization_ratio,
                  'dmtlm': dmtlm,
                  'income': income,
                  'loan_amount': loan_amount,
                  'lti': loan_amount / income if income > 0 else 0,
                  'total_loan_months': total_loan_months,
                  'loan_tenure_months': loan_tenure_months,
                  'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
                  'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
                  'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
                  'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
                  'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
                  'residence_type_Rented': 1 if residence_type == 'Rented' else 0}
    
    df = pd.DataFrame([data_input])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    df = df[features]
    
    return df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    default_probability = model.predict_proba(input_df)[:, 1]  # Probability of default
    non_default_probability = 1 - default_probability

    # Calculate the credit score based on the probabilities
    credit_score = base_score + non_default_probability.flatten() * scale_length
    
    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score), rating


def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):

    input_df = data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                         loan_amount, loan_tenure_months, total_loan_months, 
                         loan_purpose, loan_type, residence_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating