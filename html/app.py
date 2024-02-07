from flask import Flask, render_template
import pandas as pd
from pymongo import MongoClient
import json
import os
import logging

app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/")
def index():
    return "Heart Attack Prediction Data Import"

@app.route("/importData")
def importData():
    try:
        # Drop the database if it exists
        client = MongoClient('localhost', 27017)
        if 'heart_attack_prediction' in client.list_database_names():
            client.drop_database('heart_attack_prediction')
        
        # Load and process data
        file_path = "../Resources/heart_attack_prediction_dataset.csv"
        df = pd.read_csv(file_path)

        df[['Systolic Pressure', 'Diastolic Pressure']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic Pressure'] = pd.to_numeric(df['Systolic Pressure'])
        df['Diastolic Pressure'] = pd.to_numeric(df['Diastolic Pressure'])
        df.drop('Blood Pressure', axis=1, inplace=True)

        mapping = {'Female':0, 'Male':1, 'Northern Hemisphere':0, 'Southern Hemisphere':1}
        df['Sex'] = df['Sex'].map(mapping)
        df['Hemisphere'] = df['Hemisphere'].map(mapping)

        categorical_features = ['Country', 'Continent', 'Diet'] 
        categorical_dummies = pd.get_dummies(df[categorical_features])

        numerical_features = [col for col in df.columns if col not in ['Patient ID', 'Heart Attack Risk'] + categorical_features]
        encoded_df = pd.concat([df.drop(categorical_features, axis=1), categorical_dummies], axis=1)
        encoded_df['Patient ID'] = df['Patient ID']
        encoded_df = encoded_df.set_index('Patient ID')

        base_dir = os.path.abspath(os.path.dirname(__file__))
        json_file_path = os.path.join(base_dir, "Resources", "encoded_df.json")

        encoded_df.to_json(json_file_path, orient='records')

        # Insert data into MongoDB
        db = client['heart_attack_prediction']
        collection = db['heart_attack']

        with open(json_file_path) as file:
            data = json.load(file)

        collection.insert_many(data)  # Insert data into MongoDB collection

        client.close()  # Close MongoDB client connection

        # Render data in HTML template
        return render_template('index.html', data=data)
    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        return f"Error importing data: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
