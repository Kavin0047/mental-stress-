# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Student Mind Stress Calculator')

studytime = st.slider('Study Time (hours)', 1, 10, 5)
failures = st.slider('Failures', 0, 5, 0)
absences = st.slider('Absences', 0, 100, 5)
health = st.slider('Health', 1, 5, 3)

if st.button('Predict Stress Level'):
    input_data = pd.DataFrame([[studytime, failures, absences, health]], columns=['studytime', 'failures', 'absences', 'health'])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    stress_level = 'High' if prediction[0] == 1 else 'Low'
    st.write(f'The predicted stress level is: {stress_level}')
