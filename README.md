# Heart Attack Prediction Project 

## Group Members
- Mahsa Nafei
- Jesús Hernández
- Fernando López
- Alexandru Arnautu

## Overview
This repository integrates two data sources for Heart Attack Prediction through machine learning and visualization analysis. It encompasses a Flask-based web application for predicting heart attack risk by leveraging essential features and a data processing script employing Postgres SQL in Amazon RDS. Additionally, Spark is utilized for data cleaning and preparation, while data visualization and analysis are facilitated through Plotly and Folium.
 
### Objective 
This project aims to leverage the available dataset to develop a robust predictive model that effectively captures and estimates an individual's likelihood of experiencing a heart attack. Additionally, the project aims to explore the potential impact of the data analysis on informing decisions related to adjusting insurance premiums and refining underwriting processes. An essential focus is understanding the relative contribution of lifestyle factors (such as smoking, obesity, and physical activity) compared to clinical parameters (including cholesterol levels and blood pressure) in predicting heart attack risk. Furthermore, the project addresses fairness and bias, ensuring the analysis provides equitable predictions across diverse demographic groups. The ultimate goal is to enhance the accuracy of risk assessment, contribute to informed decision-making in the insurance domain, and promote fairness and transparency in evaluating heart attack risks.

#### Code Structure

### Data Preparing
- **preparing_data.ipynb** This file details the integration process of the heart attack CSV file sourced from Kaggle with data from the Countries Now API. Using Apache Spark, we conducted data cleaning and merging operations to adjust the datasets for further analysis.

### Endpoints
1. **heartattack prediction dataset**
 - URL: [https://aws-project-4.s3.ca-central-1.amazonaws.com/heart_attack_prediction_dataset.csv]
   - Description: Connection to the heart attack data set in an amazon S3 bucket.


### Data processing 
- **app.py:** The main Python script defining the Flask application, including routes for different pages and functionalities related to heart attack prediction.

- **traning_data.ipynb** This file documents our exploration of various machine-learning models for supervised and unsupervised training on the dataset. We experimented with logistic regression, random forest models, KN means, and neural networks to enhance prediction accuracy. Through a combination of techniques, including data sampling with a random oversample, grid search, and employment of a random forest classifier, we achieved a prediction accuracy of 75%. This file contains the connection to the data base. 

#### Endpoints
1. **Home Page**
   - URL: [http://localhost:5000/](http://localhost:5000/)
   - Description: Displays the heart attack app for the user to input data.
2. **results**
   - URL: [http://localhost:5000/predict](http://localhost:5000/)
   - Description: Displays the results of the heart attack application.
3. **heart_attack_prediction_db**
   - URL: [database-1.cfwmkaw8o6bp.us-east-1.rds.amazonaws.com]
   - Description: Connection to the heart attack data base in amazon RDS postgres    SQL.


### Data Analysis and Visualization

- **visuals_trial.ipynb** This file presents our response to the research questions utilizing visualizations created with the Folium library and Plotly. Through these visualizations, we offer insights and analysis to address the research question effectively.


### Data Analysis 
**here we put the sesearchquesitons and analysis**






