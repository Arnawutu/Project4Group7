from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and scaler
# Assuming you have already trained and saved the model and scaler

# Load the trained model
gs_knn = None  # Load your trained GridSearchCV model here

# Load the scaler
scaler = None  # Load your trained MinMaxScaler here

# Define the route for the home page
@app.route('/')
def home():
    return render_template('heart_attack.html')

# Define the route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cholesterol = float(request.form['cholesterol'])
        heart_rate = float(request.form['heart_rate'])
        diabetes = int(request.form['diabetes'])
        family_history = int(request.form['family_history'])
        smoking = int(request.form['smoking'])
        obesity = int(request.form['obesity'])
        alcohol_consumption = int(request.form['alcohol_consumption'])
        previous_heart_problems = int(request.form['previous_heart_problems'])
        medication_use = int(request.form['medication_use'])
        bmi = float(request.form['bmi'])
        triglycerides = float(request.form['triglycerides'])
        systolic_pressure = float(request.form['systolic_pressure'])
        diastolic_pressure = float(request.form['diastolic_pressure'])

        # Create a DataFrame with the collected input
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'Cholesterol': [cholesterol],
            'Heart Rate': [heart_rate],
            'Diabetes': [diabetes],
            'Family History': [family_history],
            'Smoking': [smoking],
            'Obesity': [obesity],
            'Alcohol Consumption': [alcohol_consumption],
            'Previous Heart Problems': [previous_heart_problems],
            'Medication Use': [medication_use],
            'BMI': [bmi],
            'Triglycerides': [triglycerides],
            'Systolic Pressure': [systolic_pressure],
            'Diastolic Pressure': [diastolic_pressure]
        })

        # Scale the input data using the loaded scaler
        scaled_input_data = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = gs_knn.predict(scaled_input_data)

        # Return the prediction result
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
