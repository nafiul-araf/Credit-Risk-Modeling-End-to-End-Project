# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:29:38 2024

@author: Admin
"""

# import os
# print(os.getcwd())

# os.chdir(r'D:\Data Science and Data Analytics\ML\Assignment - Credit Risk Modelling\Credit Risk Modelling\Project Files')

import streamlit as st
from utils import predict


# Set the page configuration and title
st.set_page_config(page_title="Credit Risk Modeling", page_icon="📊", layout="centered")
st.title("📊 Lauki Finance: Credit Risk Modelling")

# Sidebar for User Instructions
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Fill in the necessary fields on the right side.
    2. Adjust sliders and dropdowns for interactive inputs.
    3. Click 'Calculate Risk' to view results.
    """)
    st.image("project-root/Lauki Finance.JPG", caption="Your Trusted Finance Partner")  # Add a relevant image/logo.

# Input Fields
st.subheader("💼 Customer Details")

# Row 1: Age, Income, Loan Amount
col1, col2, col3 = st.columns(3)

age = col1.number_input("Age", min_value=18, max_value=100, value=28, help="Enter your age (18-100).")
income = col2.number_input("Income (Annual)", min_value=0, max_value=5000000, value=290875, step=50000, help="Your annual income in currency units.")
loan_amount = col3.number_input("Loan Amount", min_value=0, value=2560000, help="Total loan amount you want to borrow.")

# Row 2: Loan Insights
st.subheader("📊 Loan Insights")
lti = loan_amount / income if income > 0 else 0
st.metric(label="Loan-to-Income Ratio (LTI)", value=f"{lti:.2f}", help="This shows the ratio of the loan amount to your income.")

# Row 3: Loan Tenure, Avg DPD, DMTLM
st.subheader("📑 Loan Details")
col4, col5, col6 = st.columns(3)

loan_tenure_months = col4.slider("Loan Tenure (Months)", min_value=6, max_value=240, step=6, value=36, help="Select the loan tenure in months.")
avg_dpd_per_dm = col5.number_input("Avg DPD", min_value=0, value=0, help="Average Delinquent Days (Defaults), set to 0 if no loan history.")
dmtlm = col6.slider("DMTLM (Delinquent Months to Loan Ratio)", min_value=0, max_value=100, value=0, help="Delinquency ratio, 0 if no loans.")

# Row 4: Credit Utilization, Total Loan Months, Loan Purpose
st.subheader("🏡 Loan Purpose")
col7, col8, col9 = st.columns(3)

credit_utilization_ratio = col7.slider("Credit Utilization (%)", min_value=0, max_value=100, value=0, help="Percentage of utilized credit, 0 if no credit.")
total_loan_months = col8.number_input("Total Loan Months", min_value=0, value=0, help="Cumulative loan tenure across all loans, 0 if no loans.")
loan_purpose = col9.selectbox("Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'], help="Purpose of the loan.")

# Row 5: Loan Type, Residence Type
st.subheader("🏠 Loan and Residence Type")
col10, col11 = st.columns(2)

loan_type = col10.radio("Loan Type", ['Unsecured', 'Secured'], help="Choose the type of loan.")
residence_type = col11.selectbox("Residence Type", ['Owned', 'Rented', 'Mortgage'], help="Your current residence type.")

# Action Button
if st.button("Calculate Risk"):
    # Call the `predict` function with input fields
    probability, credit_score, rating = predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
                                                loan_amount, loan_tenure_months, total_loan_months,
                                                loan_purpose, loan_type, residence_type)

    # Display Results
    st.success("✅ Risk Assessment Completed!")
    st.write(f"**Default Probability:** {probability:.2%}")
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Rating:** {rating}")

    # Risk Insights
    if rating in ['Poor', 'Average']:
        st.warning("⚠ The borrower have a high-risk credit profile. Consider improving credit habits.")
    else:
        st.info("🌟 The borrower have a low-risk profile. Loan approval is likely.")
