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

- **traning_data.ipynb** This file documents our exploration of various machine-learning models for supervised  training on the dataset. We experimented with logistic regression, random forest models, KN means, and neural networks to enhance prediction accuracy. Through a combination of techniques, including data sampling with a random oversample, grid search, and employment of a random forest classifier, we achieved a prediction accuracy of 74%. This file contains the connection to the data base. 

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








#### Model Training with Original Data

Random Forest Classifier, K-Nearest Neighbors (KNN), and Logistic Regression models are trained using the original dataset without any resampling techniques. The steps involved are as follows:

1. **Data Preparation**: The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical features if necessary. This ensures that the data is in a suitable format for training the models.

2. **Feature Selection**: Relevant features are selected based on their importance using techniques such as feature importance scores, recursive feature elimination, or domain knowledge. This helps improve the models' performance by focusing on the most informative features.

3. **Model Training**: Each model (Random Forest Classifier, KNN, and Logistic Regression) is instantiated and trained using the preprocessed dataset. During training, the models learn patterns and relationships between the input features and the target variable (e.g., heart attack risk, loan status).

4. **Model Evaluation**: After training, the performance of each model is evaluated using various metrics such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC). These metrics provide insights into how well the models are able to classify instances of heart attack risk or predict loan statuses.

#### Model Training with Resampled Data

The impact of different resampling techniques on model performance is explored for each model (Random Forest Classifier, KNN, and Logistic Regression). Resampling techniques are used to address class imbalance in the dataset, where one class (e.g., low-risk of heart attack) is significantly more prevalent than the other class (e.g., high-risk of heart attack). The resampling techniques considered are:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples for the minority class to balance the class distribution.

2. **ADASYN (Adaptive Synthetic Sampling)**: Similar to SMOTE, but adjusts the synthetic samples based on the local density of minority class instances.

3. **RandomOverSampler**: Randomly samples the minority class with replacement to match the majority class's size.

4. **RandomUnderSampler**: Randomly removes samples from the majority class to match the minority class's size.

5. **ClusterCentroids**: Under-samples the majority class by clustering the data points and keeping centroids of the clusters.

For each resampling technique, the dataset is resampled, and each model (Random Forest Classifier, KNN, and Logistic Regression) is trained using the resampled data. The trained models are then evaluated using the same evaluation metrics as the models trained with the original data. This allows for a comparison of how different resampling techniques affect each model's ability to predict heart attack risk or loan statuses accurately.





#### Setup

Ensure that you have the necessary Python libraries installed, including Pandas, NumPy, scikit-learn, psycopg2, and imbalanced-learn. Additionally, make sure you have access to a PostgreSQL database where the heart attack prediction dataset is stored.

#### Instructions

1. Ensure you have access to the PostgreSQL database containing the heart attack prediction dataset.

2. Install the required Python libraries mentioned in the setup section.

3. Execute the code cells sequentially in your preferred Python environment (e.g., Jupyter Notebook) to perform data retrieval, preprocessing, model training, and evaluation.

4. Review the results and analysis provided to understand the predictive performance of the models and make informed decisions regarding heart attack risk prediction.




