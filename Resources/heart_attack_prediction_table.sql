CREATE TABLE HeartAttackPrediction (
    ID TEXT PRIMARY KEY,
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
);

SELECT * FROM HeartAttackPrediction;

