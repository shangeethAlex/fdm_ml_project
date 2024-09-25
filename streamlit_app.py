import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Function to make predictions
def predict(data):
    pred_df = data.get_data_as_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return results[0]

# Streamlit app
st.set_page_config(page_title="Student Exam Performance Prediction", layout="centered")

st.title("Student Exam Performance Prediction")

# Input fields
gender = st.selectbox("Gender", ["Select your Gender", "male", "female"])
ethnicity = st.selectbox("Race or Ethnicity", ["Select Ethnicity", "group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education", ["Select Parent Education", "associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
lunch = st.selectbox("Lunch Type", ["Select Lunch Type", "free/reduced", "standard"])
test_preparation_course = st.selectbox("Test Preparation Course", ["Select Test Course", "none", "completed"])
reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100, step=1)

# Predict button
if st.button("Predict your Math Score"):
    if gender != "Select your Gender" and ethnicity != "Select Ethnicity" and parental_level_of_education != "Select Parent Education" and lunch != "Select Lunch Type" and test_preparation_course != "Select Test Course":
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )
        prediction = predict(data)
        st.success(f"The predicted Math score is: {prediction}")
    else:
        st.error("Please fill in all the required fields.")

