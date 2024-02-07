from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and scaler
# Load the data
encoded_df = pd.read_json("https://aws-project-4.s3.ca-central-1.amazonaws.com/encoded_df.json")

# Separate features and target variable
X = encoded_df.drop(['Exercise Hours Per Week',
     'Stress Level',
       'Sedentary Hours Per Day', 'Income', 
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Hemisphere',
       'Heart Attack Risk',
       'Country_Argentina', 'Country_Australia', 'Country_Brazil',
       'Country_Canada', 'Country_China', 'Country_Colombia', 'Country_France',
       'Country_Germany', 'Country_India', 'Country_Italy', 'Country_Japan',
       'Country_New Zealand', 'Country_Nigeria', 'Country_South Africa',
       'Country_South Korea', 'Country_Spain', 'Country_Thailand',
       'Country_United Kingdom', 'Country_United States', 'Country_Vietnam',
       'Continent_Africa', 'Continent_Asia', 'Continent_Australia',
       'Continent_Europe', 'Continent_North America',
       'Continent_South America', 'Diet_Average', 'Diet_Healthy',
       'Diet_Unhealthy'], axis=1)
y = encoded_df['Heart Attack Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating StandardScaler instance
scaler = MinMaxScaler()

# Fitting StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the trained model
# Define the pipeline including MinMaxScaler and LogisticRegression
log_reg_pipe = Pipeline([
    ('mms', MinMaxScaler()),
    ('log_reg', LogisticRegression())
])

# Define the parameters grid for GridSearchCV
params = {
    'log_reg__C': np.logspace(-3, 3, 7),
    'log_reg__penalty': ['l2']  # Update to include only 'l2' penalty
}

# Initialize GridSearchCV with the pipeline, parameters, scoring, and cross-validation
gs_log_reg = GridSearchCV(
    log_reg_pipe,
    param_grid=params,
    scoring='accuracy',
    cv=10
)

# Fit GridSearchCV on the training data
gs_log_reg.fit(X_train_scaled, y_train)


# Define the route for the home page
@app.route('/')
def home():
    return render_template('heart_attack.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cholesterol = int(request.form['cholesterol'])
        heart_rate = int(request.form['heart_rate'])
        diabetes = int(request.form['diabetes'])
        family_history = int(request.form['family_history'])
        smoking = int(request.form['smoking'])
        obesity = int(request.form['obesity'])
        alcohol_consumption = int(request.form['alcohol_consumption'])
        previous_heart_problems = int(request.form['previous_heart_problems'])
        medication_use = int(request.form['medication_use'])
        bmi = float(request.form['bmi'])
        triglycerides = int(request.form['triglycerides'])
        systolic_pressure = int(request.form['systolic_pressure'])
        diastolic_pressure = int(request.form['diastolic_pressure'])

        # Create a feature vector from the user input
        user_input = [[age, sex, cholesterol, heart_rate, diabetes, family_history, smoking, obesity,
                    alcohol_consumption, previous_heart_problems, medication_use, bmi, triglycerides,
                    systolic_pressure, diastolic_pressure]]

        # Scale the feature vector using the scaler fitted on the training data
        user_input_scaled = scaler.transform(user_input)

        # Make a prediction
        prediction = gs_log_reg.predict(user_input_scaled)

        # Determine the prediction message
        if prediction == 1:
            prediction_message = "High risk of heart attack"
        else:
            prediction_message = "Low risk of heart attack"

        # Get the test data accuracy
        test_accuracy = round(gs_log_reg.score(X_test_scaled, y_test) * 100, 2)


        # Render the prediction result page with the prediction message and test accuracy
        return render_template('result.html', prediction_message=prediction_message, test_accuracy=test_accuracy)


if __name__ == '__main__':
    app.run(debug=True)