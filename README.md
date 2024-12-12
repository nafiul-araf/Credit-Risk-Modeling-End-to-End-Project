# Credit Risk Modeling

![image](https://github.com/user-attachments/assets/e6c7a247-fc29-487a-8bf8-a32907b638d2)


## Project Overview
Credit risk modeling is crucial for financial institutions to assess the likelihood of a borrower defaulting on a loan. This project involves analyzing multiple datasets to identify factors influencing credit risk, ultimately leading to better decision-making processes.

[Web Link](https://credit-risk-modeling-lauki-finance.streamlit.app/)

## Default Risk Prediction: Model Evaluation and Deployment

### Overview
This project aims to develop a machine learning model to predict default risk, ensuring high accuracy and interpretability. The final model leverages advanced techniques to provide actionable insights, making it suitable for real-world deployment.

### Key Features
- **Dataset**: Imbalanced classification problem with 10% defaults.
- **Techniques**:
  - Feature engineering using domain relevance and statistical analysis.

    [FI](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/Feature%20importance.png)

  - Resampling methods (over-sampling via SMOTE, under-sampling).
- **Models Evaluated**:
  - Logistic Regression
  - Random Forest
  - XGBoost

### Selected Model
- **Model**: XGBoost with Optuna hyperparameter tuning and under-sampling.
- **Metrics**:
  - AUC: 0.98
  - Gini Coefficient: 0.97
  - KS Statistic: 86.87%
- **Interpretability Tools**:
  - SHAP (feature importance)
  - LIME (local interpretability)

    [lime](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/Lime.JPG)

### Key Results
- The model demonstrates superior ability to classify defaults with high precision and recall.
- Decile analysis confirms excellent separation of high-risk instances.

#### Deployment Readiness
- **Strengths**:
  - High performance across metrics
  - Interpretability ensures alignment with business and regulatory requirements.
- **Mitigation Strategies**: Address risks from under-sampling by periodic retraining.

### Visualizations
1. AUC-ROC curve with near-perfect performance (AUC: 0.99).

   [rocauc](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/ROC%20Curve.png)
   
2. SHAP summary plot illustrating top features influencing predictions.

### How to Use
1. **Train the Model**: Scripts for data preprocessing, training, and hyperparameter tuning are included.
2. **Evaluate the Model**: Tools for generating metrics, decile analysis, and interpretability plots.
3. **Deploy the Model**: Prebuilt deployment pipeline for integration into business systems.

### Why This Project Stands Out
- Combines state-of-the-art machine learning techniques with interpretability.
- Addresses a real-world business problem with rigor and precision.
- Provides a clear path from model development to deployment.

---

#### Get Involved
Feel free to clone the repository and contribute to further enhancements!

