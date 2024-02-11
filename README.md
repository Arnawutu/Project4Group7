# Heart Attack Prediction Project 

## Group Members
- Mahsa Nafei
- Jesús Hernández
- Fernando López
- Alexandru Arnautu

## Overview
Our repository combines two data sources and API Coordinates to predict heart attack risk using machine learning and visualization analysis. It features a Flask-based web application for risk prediction, a data processing script using Postgresql in Amazon RDS, and employs Spark for data cleaning and preparation. We employed data Plotly, Pandas and Folium for visualization and analysis.
 
### Objective 
This project focuses on using a dataset to create a predictive model for assessing an individual's risk of having a heart attack. It aims to understand how specific  lifestyle factors influence the probability of heart risk

- Age
- Blood Pressure
- Cholesterol Levels 
- Diabetic Status
- Hours of Sleep
- Medication Use
- Obesity
- Physical Activity 
- Smoking 
- Triglycerides

Our project seeks to examine the data outlined above to help insurance companies provide  more detail specific health programs for patients affected by heart risk, by looking at related factors. A critical aspect of the project is ensuring fairness and reducing bias in predictions across different demographic groups, and instead focused on creating a global pattern rather than continent/country specific ones. Our goals throughout the making of this analysis, was to focus on  improving risk assessment accuracy through supervised machine learning to support informed decision-making in heart attack risk evaluation.


#### Questions
- Health Risk assessment
  - Question: Based on the available data, how effectively does the model capture and estimate an individual's likelihood of experiencing a heart attack?
  - Question: Which feature in our dataset impacts the possibility of a heart attack the most?

- Fairness and Bias:
  - Question: How does the analysis account for fairness and avoid bias in predicting heart attack risks across different demographic groups?
  - Question: What is the age distribution of the heart attack risk in the data set?
  - Question: What is the distribution by country and continent of heart attack risk?

- Impact of Lifestyle Factors on Risk Assessment:
  - Question: What is the relative contribution of lifestyle factors (e.g., smoking, obesity, physical activity, hours of sleep) versus clinical parameters (e.g., cholesterol levels, triglycerides, blood pressure) in predicting heart attack risk?
  - Question: How does the amount of sleep and physical activity influence the risk of an individual becoming susceptible to heart attack?
  - Question: How does triglycerides and cholesterol levels impact the risk of heart attack at a continental scale? 

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




