import streamlit as st
import pandas as pd
import joblib

# Load the saved model
loaded_model = joblib.load("best_model_rf.joblib")

# Create the Streamlit app
st.title("Diabetes Prediction App")
st.markdown("---")

# Input options for prediction
st.subheader("Prediction Options")
age_input = st.slider("Age", 20, 100, 50)
gender_input = st.radio("Gender", ["Male", "Female"])
hypertension_input = st.checkbox("Hypertension")
heart_disease_input = st.checkbox("Heart Disease")
bmi_input = st.number_input("BMI", value=25.0)
hba1c_level_input = st.number_input("HbA1c Level", value=6.0)
blood_glucose_input = st.number_input("Blood Glucose Level", value=120)

# Convert the input to the format expected by the model
gender_mapping = {'Male': 0, 'Female': 1}
gender_encoded = gender_mapping[gender_input]

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'age': [age_input],
    'gender': [gender_encoded],
    'hypertension': [int(hypertension_input)],
    'heart_disease': [int(heart_disease_input)],
    'bmi': [bmi_input],
    'HbA1c_level': [hba1c_level_input],
    'blood_glucose_level': [blood_glucose_input]
})

# Add a "Predict" button to trigger the prediction
predict_button = st.button("Predict")

# Perform prediction when the "Predict" button is clicked
if predict_button:
    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data)

    # Display the prediction result with larger font and red color
    st.markdown(
        f"""
        <div style='font-size: 24px; color: red;'>
            <p>Prediction Result: {'No Diabetes' if prediction[0] == 0 else 'Diabetes'}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add a watermark to the app
st.markdown(
    """
    <div style='position: absolute; top: 20px; right: 20px; color: #999;'>
        Made by: Abdelrahman Eldaba
    </div>
    """,
    unsafe_allow_html=True
)