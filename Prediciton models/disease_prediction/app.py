import streamlit as st
import sys
sys.path.append('Prediciton models/disease_prediction/code')

from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array


# Create disease class and load ML model
disease_model = DiseaseModel()
disease_model.load_xgboost('model/xgboost_model.json')

# Set page width to wide
st.set_page_config(layout='wide')

def app():
    """Main application function."""
    # Sidebar content
    st.sidebar.title("Health Prediction")
    st.sidebar.write("Select your symptoms from the list below:")

    # Title
    st.write('## General Health Prediction')

    # User input for symptoms
    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    # Prepare symptoms array
    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'):
        # Run the model with the python script
        prediction, prob = disease_model.predict(X)
        st.write(f'### Disease: {prediction} with {prob*100:.2f}% probability')

        # Create tabs for Description and Precautions
        tab1, tab2 = st.tabs(["Description", "Precautions"])

        with tab1:
            # Show disease description
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            # Show precautions
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')

             

