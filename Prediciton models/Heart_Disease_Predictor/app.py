import streamlit as st
import pickle
import pandas as pd

# Load the saved model
with open("DPrediciton models/Heart_Disease_Predictor/logistic_model_pipeline.pkl2", "rb") as f:
    pipeline = pickle.load(f)

# Define a function to make predictions
def predict_heart_disease(age, sex_male, cigs_per_day, tot_chol, sys_bp, glucose):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'Sex_male': [sex_male],
        'cigsPerDay': [cigs_per_day],
        'totChol': [tot_chol],
        'sysBP': [sys_bp],
        'glucose': [glucose]
    })
    
    # Make prediction using the loaded model
    prob = pipeline.predict_proba(input_data)[:, 1][0]  # Probability of having heart disease
    return prob * 100  # Return the probability as a percentage

# Define the app function
def app():
    # Streamlit app layout
    st.markdown("<h1 style='font-size: 36px;'>Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 24px;'>Check your risk of heart disease based on your details.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 20px;'>Fill in the information below to predict your heart disease risk.</p>", unsafe_allow_html=True)

    # Use columns to display inputs side by side with padding
    col1, col2 = st.columns(2)

    # Age input in the first column
    with col1:
        age = st.slider("Age", min_value=20, max_value=100, value=50, step=1, help="Enter your age (20-100).", key="age", label_visibility="visible")

    # Sex input in the second column
    with col2:
        sex_male = st.radio("Sex", options=[1, 0], index=1, format_func=lambda x: "Male" if x == 1 else "Female", help="Select your sex: 1 for Male, 0 for Female.", key="sex_male")

    # Cigarettes per day input in the first column
    with col1:
        cigs_per_day = st.slider("Cigarettes per Day", min_value=0, max_value=60, value=10, step=1, help="Enter the number of cigarettes you smoke per day (0-60).", key="cigs_per_day")

    # Total cholesterol input in the second column
    with col2:
        tot_chol = st.slider("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1, help="Enter your total cholesterol level (100-400 mg/dL).", key="tot_chol")

    # Systolic blood pressure input in the first column
    with col1:
        sys_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=90, max_value=200, value=120, step=1, help="Enter your systolic blood pressure (90-200 mmHg).", key="sys_bp")

    # Glucose input in the second column
    with col2:
        glucose = st.slider("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, step=1, help="Enter your glucose level (50-300 mg/dL).", key="glucose")

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction button centered
    if st.button("Predict", key="predict", help="Click to predict the probability of heart disease"):
        prob = predict_heart_disease(age, sex_male, cigs_per_day, tot_chol, sys_bp, glucose)
        st.success(f"The probability of having heart disease is: **{prob:.2f}%**", icon="✅")
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # Display a fun message based on the probability
        if prob < 30:
            st.write("<p style='font-size: 20px;'>You're in the low-risk category! Keep up the good work with your health.</p>", unsafe_allow_html=True)
        elif prob < 70:
            st.write("<p style='font-size: 20px;'>Your risk is moderate. Consider adopting a healthier lifestyle.</p>", unsafe_allow_html=True)
        else:
            st.write("<p style='font-size: 20px;'>Your risk is high. Please consult a healthcare professional for further advice.</p>", unsafe_allow_html=True)

        # Add a horizontal line and spacing
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

