from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

# Initialize Flask application
app = Flask(__name__)

############# START POSTGRES CONNECTION ############
# Info for the connection with PostgreSQL in RDS
PGEND_POINT = 'database-1.cfwmkaw8o6bp.us-east-1.rds.amazonaws.com'  # End Point
PGDATABASE_NAME = 'heart_attack_prediction_db'  # Database name
PGUSER_NAME = 'postgres'
PGPASSWORD = 'B00TC4MP'


def connect():
    conn_string = f"host={PGEND_POINT} port=5432 dbname={PGDATABASE_NAME} user={PGUSER_NAME} password={PGPASSWORD}"
    conn = psycopg2.connect(conn_string)
    print("Connected!")
    cursor = conn.cursor()
    return conn, cursor


def close_connection(conn, cursor):
    conn.commit()
    cursor.close()
    conn.close()
    print("Connection closed.")


conn, cursor = connect()

# SQL SELECT statement to retrieve all columns (*) from the table named encodedtable2
query_hat_all = sql.SQL("""
SELECT * FROM encodedtable2;
""")
cur = conn.cursor()
cur.execute(query_hat_all)

# Put all the data in the encodedtable2 table into a DataFrame in pandas
encoded_df = pd.DataFrame(cur.fetchall(), columns=['Patient ID', 'Country', 'Capital', 'Age', 'Sex', 'Cholesterol',
                                                   'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
                                                   'Alcohol Consumption', 'Exercise Hours Per Week',
                                                   'Previous Heart Problems', 'Medication Use', 'Stress Level',
                                                   'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                                                   'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Continent',
                                                   'Hemisphere', 'Heart Attack Risk', 'Systolic Pressure',
                                                   'Diastolic Pressure', 'lat', 'long', 'Diet_Average', 'Diet_Healthy',
                                                   'Diet_Unhealthy'])

close_connection(conn, cursor)
############ END POSTGRES CONNECTION ############

# Separate features and target variable
X = encoded_df.drop(['Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day', 'Income',
                     'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Heart Attack Risk', 'Diet_Average',
                     'Diet_Healthy', 'Diet_Unhealthy', 'Country', 'Capital', 'lat', 'long', 'Continent',
                     'Patient ID', 'Hemisphere'], axis=1)
y = encoded_df['Heart Attack Risk']

# Create an instance of RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Resample the data
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Normalize the feature matrix X_resampled
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42)

# Normalize the feature matrix after splitting
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the random forest classifier with default hyperparameters
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the entire training set with the best hyperparameters
best_rf_model = RandomForestClassifier(max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300,
                                        random_state=42)
best_rf_model.fit(X_train_scaled, y_train)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('heart_attack.html')

# Define a route for the map
@app.route('/cholesterol_triglycerides_map')
def cholesterol_triglycerides_map():
    # Assuming heart_trichol is a DataFrame with the required data
    heart_trichol = pd.read_csv("path/to/heart_trichol.csv")  # Replace with the actual path

    my_map = folium.Map(location=[27.2546, 33.8116], zoom_start=2.3)

    # Add markers for each city with popup showing cholesterol and triglycerides information
    for city, lat in heart_trichol['lat'].items():
        long = heart_trichol['long'][city]
        cholesterol = heart_trichol['Cholesterol'][city]
        triglycerides = heart_trichol['Triglycerides'][city]

        popup_content = f'<b>{city}</b><br>Cholesterol: {cholesterol}<br>Triglycerides: {triglycerides}'
        folium.Marker(location=[lat, long], popup=popup_content).add_to(my_map)

    # Save the map to an HTML file
    map_filename = "static/cholesterol_triglycerides_map.html"  # Adjust the path as needed
    my_map.save(map_filename)

    # Render the template with the map
    return render_template('cholesterol_triglycerides_map.html', map_filename=map_filename)

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
        prediction = best_rf_model.predict(user_input_scaled)

        # Determine the prediction message
        if prediction == 1:
            prediction_message = "High risk of heart attack"
        else:
            prediction_message = "Low risk of heart attack"

        # Get the test data accuracy
        test_accuracy = round(best_rf_model.score(X_test_scaled, y_test) * 100, 2)


        # Render the prediction result page with the prediction message and test accuracy
        return render_template('result.html', prediction_message=prediction_message, test_accuracy=test_accuracy)


if __name__ == '__main__':
    app.run(debug=True)
