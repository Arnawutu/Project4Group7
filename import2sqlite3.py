import sqlite3
import pandas as pd

from pathlib import Path

database_path = "Resources/heart_attack.sqlite"
Path(database_path).touch()

conn = sqlite3.connect(database_path)
c = conn.cursor()

# c.execute('''CREATE TABLE HeartAttackPrediction (Patient ID string,	Age int, Sex string, Cholesterol int, BloodPressure string, 
# HeartRate int, Diabetes int, FamilyHistory int, Smoking int, Obesity int, AlcoholConsumption int, ExerciseHoursPerWeek float,
# Diet string, PreviousHeartProblems int, MedicationUse int, StressLevel int,SedentaryHoursPerDay float, 
# Income int,	BMI float, Triglycerides int, PhysicalActivityDaysPerWeek int,SleepHoursPerDay int, Country string,	
# Continent string, Hemisphere string, HeartAttackRisk int)''')

c.execute('''CREATE TABLE HeartAttackPrediction (
                Patient_ID TEXT,
                Age INTEGER,
                Sex TEXT,
                Cholesterol INTEGER,
                BloodPressure TEXT,
                HeartRate INTEGER,
                Diabetes INTEGER,
                FamilyHistory INTEGER,
                Smoking INTEGER,
                Obesity INTEGER,
                AlcoholConsumption INTEGER,
                ExerciseHoursPerWeek REAL,
                Diet TEXT,
                PreviousHeartProblems INTEGER,
                MedicationUse INTEGER,
                StressLevel INTEGER,
                SedentaryHoursPerDay REAL,
                Income INTEGER,
                BMI REAL,
                Triglycerides INTEGER,
                PhysicalActivityDaysPerWeek INTEGER,
                SleepHoursPerDay INTEGER,
                Country TEXT,
                Continent TEXT,
                Hemisphere TEXT,
                HeartAttackRisk INTEGER
            )''')



csv_icecream = pd.read_csv("Resources/heart_attack_prediction_dataset.csv")
csv_icecream.to_sql("HeartAttackPrediction", conn, if_exists='append', index=False)
conn.close()
