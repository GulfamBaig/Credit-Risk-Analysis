import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import os

# Set page config
st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        .stNumberInput>div>div>input {
            border-radius: 5px;
        }
        .stSelectbox>div>div>select {
            border-radius: 5px;
        }
        .title {
            color: #2c3e50;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .result-box {
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .risk-meter {
            height: 30px;
            width: 100%;
            background: #ddd;
            border-radius: 5px;
            margin: 10px 0;
            position: relative;
        }
        .risk-meter-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        .risk-label {
            position: absolute;
            width: 100%;
            text-align: center;
            line-height: 30px;
            color: white;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }
        .info-box {
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 5px;
            border-left: 5px solid #4CAF50;
        }
        .feature-card {
            background: white;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature-card h4 {
            margin-top: 0;
            color: #2c3e50;
        }
        .feature-card p {
            margin-bottom: 0;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Check if required files exist
        required_files = {
            'model': 'credit_risk_model.pkl',
            'scaler': 'scaler.pkl',
            'imputer_median': 'imputer_median.pkl',
            'imputer_mode': 'imputer_mode.pkl',
            'feature_names': 'feature_names.pkl'
        }
        
        missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

        # Load XGBoost model from pickle file
        with open(required_files['model'], 'rb') as f:
            model = pickle.load(f)
        
        # Verify model type
        if not isinstance(model, XGBClassifier):
            raise TypeError("Loaded model is not an XGBClassifier")
        
        # Load other preprocessing artifacts
        scaler = joblib.load(required_files['scaler'])
        imputer_median = joblib.load(required_files['imputer_median'])
        imputer_mode = joblib.load(required_files['imputer_mode'])
        
        with open(required_files['feature_names'], 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, scaler, imputer_median, imputer_mode, feature_names
        
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.error("Please ensure all these files exist in your directory:")
        st.error("- credit_risk_model.pkl (XGBoost model)")
        st.error("- scaler.pkl (StandardScaler)")
        st.error("- imputer_median.pkl (SimpleImputer for numerical features)")
        st.error("- imputer_mode.pkl (SimpleImputer for categorical features)")
        st.error("- feature_names.pkl (list of feature names)")
        st.stop()

model, scaler, imputer_median, imputer_mode, feature_names = load_artifacts()

# Feature engineering function
def engineer_features(input_df):
    df = input_df.copy()
    # Create new features
    df["TotalMissedPayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )
    df["IncomeDebtRatio"] = np.where(
        df["DebtRatio"] == 0, 
        0, 
        df["MonthlyIncome"] / (df["DebtRatio"] * df["MonthlyIncome"] + 1e-6)
    )
    df["CreditBurden"] = df["RevolvingUtilizationOfUnsecuredLines"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    
    # Age group feature
    df["age"] = df["age"].astype(int)
    df["AgeGroup"] = pd.cut(
        df["age"], 
        bins=[0, 30, 50, 65, 100], 
        labels=["Young", "Middle-aged", "Senior", "Elderly"]
    )
    return df

# Preprocess input data
def preprocess_input(input_df):
    try:
        # Impute missing values
        input_df["MonthlyIncome"] = imputer_median.transform(input_df[["MonthlyIncome"]])
        input_df["NumberOfDependents"] = imputer_mode.transform(input_df[["NumberOfDependents"]])
        
        # Feature engineering
        processed_df = engineer_features(input_df)
        
        # Convert categorical features
        processed_df = pd.get_dummies(processed_df, columns=["AgeGroup"], drop_first=True)
        
        # Ensure all expected columns are present
        for col in feature_names:
            if col not in processed_df.columns and col != "SeriousDlqin2yrs":
                processed_df[col] = 0
        
        # Reorder columns to match training data
        processed_df = processed_df[feature_names]
        
        # Scale features
        scaled_data = scaler.transform(processed_df)
        
        return scaled_data
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        st.stop()

# Main app function
def main():
    st.title("💰 Credit Risk Analysis")
    st.markdown("""
    Predict the probability of a borrower experiencing serious financial distress in the next 2 years.
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Borrower Information")
        
        # Input form
        with st.form("credit_form"):
            age = st.number_input("Age", min_value=18, max_value=100, value=30, 
                                help="Borrower's age in years")
            
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000,
                                           help="Borrower's monthly income")
            
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.5, step=0.01, 
                                       format="%.2f", help="Monthly debt payments divided by monthly income")
            
            revol_util = st.number_input("Revolving Line Utilization Rate", min_value=0.0, max_value=1.0, 
                                       value=0.5, step=0.01, format="%.2f",
                                       help="Amount of credit the borrower is using relative to available credit")
            
            num_open_credit = st.number_input("Number of Open Credit Lines/Loans", min_value=0, value=5,
                                            help="Total number of open credit lines and loans")
            
            num_real_estate = st.number_input("Number of Real Estate Loans/Lines", min_value=0, value=1,
                                            help="Number of mortgages and real estate loans")
            
            num_dependents = st.number_input("Number of Dependents", min_value=0, value=0,
                                           help="Number of dependents in family (excluding self)")
            
            st.subheader("Payment History")
            late_30_59 = st.number_input("30-59 Days Past Due (Count)", min_value=0, value=0,
                                       help="Number of times borrower has been 30-59 days past due")
            
            late_60_89 = st.number_input("60-89 Days Past Due (Count)", min_value=0, value=0,
                                       help="Number of times borrower has been 60-89 days past due")
            
            late_90 = st.number_input("90+ Days Past Due (Count)", min_value=0, value=0,
                                    help="Number of times borrower has been 90+ days past due")
            
            submitted = st.form_submit_button("Predict Credit Risk")
    
    with col2:
        if submitted:
            # Create input dataframe
            input_data = {
                "RevolvingUtilizationOfUnsecuredLines": revol_util,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": num_open_credit,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": num_real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfDependents": num_dependents
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Preprocess and predict
            with st.spinner("Analyzing credit risk..."):
                processed_data = preprocess_input(input_df)
                probability = model.predict_proba(processed_data)[0, 1]
            
            # Display results
            st.subheader("Risk Assessment")
            
            with st.container():
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                
                # Risk level interpretation
                if probability < 0.3:
                    risk_level = "Low Risk"
                    color = "#4CAF50"
                    recommendation = "✅ This applicant appears to be a low credit risk."
                elif probability < 0.7:
                    risk_level = "Medium Risk"
                    color = "#FFA500"
                    recommendation = "⚠️ This applicant has moderate credit risk. Further review recommended."
                else:
                    risk_level = "High Risk"
                    color = "#F44336"
                    recommendation = "❌ This applicant appears to be a high credit risk."
                
                st.markdown(f"""
                <h3 style='color:{color}; text-align:center;'>{risk_level}</h3>
                <p style='text-align:center; font-size:18px;'>Probability of Serious Delinquency: <b>{probability:.1%}</b></p>
                """, unsafe_allow_html=True)
                
                # Risk meter
                st.markdown(f"""
                <div class='risk-meter'>
                    <div class='risk-meter-fill' style='width:{probability*100}%; background:{color};'></div>
                    <div class='risk-label'>{probability*100:.1f}%</div>
                </div>
                <p style='text-align:center;'>{recommendation}</p>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Key factors explanation
                st.markdown("""
                <div class='info-box'>
                    <h4>Key Risk Factors</h4>
                    <div class='feature-card'>
                        <h4>Payment History</h4>
                        <p>Late payments significantly impact credit risk. Multiple late payments increase risk.</p>
                    </div>
                    <div class='feature-card'>
                        <h4>Debt-to-Income Ratio</h4>
                        <p>Higher ratios indicate greater financial strain and higher risk.</p>
                    </div>
                    <div class='feature-card'>
                        <h4>Credit Utilization</h4>
                        <p>Using a high percentage of available credit can negatively impact risk assessment.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.subheader("Risk Assessment")
            st.markdown("""
            <div class='result-box' style='text-align:center; padding:40px;'>
                <p>Please fill out the borrower information and click <b>'Predict Credit Risk'</b> to see results.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Information about credit risk
            st.markdown("""
            <div class='info-box'>
                <h4>About Credit Risk Analysis</h4>
                <p>This model predicts the probability that a borrower will experience serious financial distress in the next 2 years.</p>
                
                <div class='feature-card'>
                    <h4>Risk Categories</h4>
                    <p><b>Low Risk</b> (0-30% probability)</p>
                    <p><b>Medium Risk</b> (30-70% probability)</p>
                    <p><b>High Risk</b> (70-100% probability)</p>
                </div>
                
                <p><b>Note:</b> This tool provides predictive analytics to support, not replace, credit decisions.</p>
            </div>
            """, unsafe_allow_html=True)

    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:14px;'>
        <p>Credit Risk Analysis Model | Built with XGBoost | v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
