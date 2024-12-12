# Credit Risk Modeling

## Project Overview
Credit risk modeling is crucial for financial institutions to assess the likelihood of a borrower defaulting on a loan. This project involves analyzing multiple datasets to identify factors influencing credit risk, ultimately leading to better decision-making processes. The project provides a credit risk assessment system powered by machine learning. It evaluates borrowers' default risk, calculates credit scores, and assigns credit ratings. The project is built using Python and Streamlit, providing an interactive and user-friendly interface.

[**Web Link**](https://credit-risk-modeling-lauki-finance.streamlit.app/)

## Default Risk Prediction: Model Evaluation and Deployment

### Overview
This project aims to develop a machine learning model to predict default risk, ensuring high accuracy and interpretability. The final model leverages advanced techniques to provide actionable insights, making it suitable for real-world deployment.

### Key Features
- **Dataset**: Imbalanced classification problem with 10% defaults.
- **Techniques**:
  - Feature engineering using domain relevance and statistical analysis.
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

    ![FI](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/Feature%20importance.png)

  - LIME (local interpretability)

    ![lime](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/Lime.JPG)

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

   ![rocauc](https://github.com/nafiul-araf/Credit-Risk-Modeling-End-to-End-Project/blob/main/images/ROC%20Curve.png)
   
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



# **Running the Project: Credit Risk Modeling**

## **Features**
- **Interactive Credit Risk Assessment**: Input borrower and loan details and get real-time predictions.
- **Advanced Machine Learning**: Uses a fine-tuned XGBoost model for robust and accurate predictions.
- **Scalable Design**: Modular structure with reusable utilities and hyperparameter tuning.

---

## **Project Directory Structure**

```
project-root/
│
├── model/
│   ├── model_data.pkl                # Serialized machine learning model and preprocessing data
│   ├── tuned_hyperparameters.txt    # Details of the optimized hyperparameters
│
├── Lauki Finance.JPG                # Project logo or related image
├── Readme.md                        # Documentation file
├── main.py                          # Streamlit application file
├── requirements.txt                 # List of required Python packages
├── utils.py                         # Utility functions for prediction and preprocessing
```

---

## **Installation Guide**

### **Step 1: Clone the Repository**
Download the project repository to your local machine:
```bash
git clone https://github.com/username/repository-name.git
cd repository-name//project-root
```

### **Step 2: Set Up the Python Environment**
Ensure you have Python 3.8 or higher installed. It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### **Step 3: Install Dependencies**
Install all the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### **Step 4: Run the Application**
Start the Streamlit application by running the following command:
```bash
streamlit run main.py
```

---

## **How to Use**
1. Open the URL displayed in your terminal after running `streamlit run main.py`. Typically, it will be something like `http://localhost:8501/`.
2. Use the interactive interface to:
   - Enter borrower details (age, income, loan amount, etc.).
   - Adjust sliders and dropdowns for other inputs.
   - Click "Calculate Risk" to view the results, including:
     - Default Probability
     - Credit Score
     - Credit Rating
3. Review the risk insights and recommendations provided in the results.

---

## **Additional Notes**

### **Dependencies**
The project requires the following key libraries:
- `streamlit`: For building the interactive web interface.
- `scikit-learn`: For preprocessing and model handling.
- `joblib`: For loading the serialized model.
- `pandas` and `numpy`: For data manipulation.
- `xgboost` and others

All dependencies are listed in `requirements.txt` for easy installation.

### **Customizations**
- To use a different machine learning model, replace `model_data.pkl` with your serialized model and adjust the features in `utils.py`.
- Update the interface in `main.py` to reflect any changes to the inputs or outputs.

---

## **Example Screenshots**
1. **Home Page**: Displays the project title and input interface.
2. **Results Page**: Shows default probability, credit score, and rating with actionable insights.

![image](https://github.com/user-attachments/assets/72691a32-1530-474f-8d87-fd43b0aab52b)

---



