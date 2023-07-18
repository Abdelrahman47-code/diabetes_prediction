import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("diabetes_prediction.csv")

# For simplicity, we'll just drop the "smoking_history" column and convert "gender" to numerical value
df = df.drop("smoking_history", axis=1)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Split the data into features (X) and the target variable (y)
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Use a SimpleImputer to fill missing values with the mean of the column
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for data augmentation
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Create the Streamlit app
st.title("Diabetes Prediction App")

# Display model accuracy after hyperparameter tuning
st.subheader("Model Selection and Hyperparameter Tuning")
st.write("Best Model: Random Forest")
st.write(f"Best Accuracy: {grid_search.best_score_:.2f}")

# Display confusion matrix for the best model
st.subheader("Confusion Matrix")
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Display classification report for the best model
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred_best, target_names=["No Diabetes", "Diabetes"]))

# Input options for prediction
st.sidebar.title("Prediction Options")
age_input = st.sidebar.slider("Age", 20, 100, 50)
gender_input = st.sidebar.radio("Gender", ["Male", "Female"])
hypertension_input = st.sidebar.checkbox("Hypertension")
heart_disease_input = st.sidebar.checkbox("Heart Disease")
bmi_input = st.sidebar.number_input("BMI", value=25.0)
hba1c_level_input = st.sidebar.number_input("HbA1c Level", value=6.0)
blood_glucose_input = st.sidebar.number_input("Blood Glucose Level", value=120)

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
predict_button = st.sidebar.button("Predict")

# Perform prediction when the "Predict" button is clicked
if predict_button:
    # Make predictions using the loaded model
    prediction = best_model.predict(input_data)

    # Display the prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.write("Prediction: No Diabetes")
    else:
        st.write("Prediction: Diabetes")
