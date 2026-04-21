import numpy as np
import pandas as pd

df = pd.read_csv("asthma_disease_data_new.csv")

def Overview():
    whole_missing_rows = df[df.isna().all(axis=1)]
    result = len(whole_missing_rows)
    print(f"Detected whole missing rows : {result}")

def fillID():
    start_id = 5034
    new_id = np.arange(start_id, start_id + len(df))
    df["PatientID"] = new_id
    print("Process 1 : Filling PatientID completed")

def fillMed():
    num_cols = ["Age", "BMI", "PhysicalActivity", "DietQuality",
            "SleepQuality", "PollutionExposure", "PollenExposure",
            "DustExposure", "LungFunctionFEV1", "LungFunctionFVC"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    print("Process 2 : Fill median value in numerical columns completed")

def fillMode():
    catego_cols = ["Gender", "Smoking", "PetAllergy",
            "FamilyHistoryAsthma", "HistoryOfAllergies",
            "Eczema", "HayFever", "GastroesophagealReflux",
            "Wheezing", "ShortnessOfBreath", "ChestTightness",
            "Coughing", "NighttimeSymptoms", "ExerciseInduced",
            "Diagnosis", "EducationLevel", "Ethnicity"]
    for col in catego_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("Process 3 : Fill mode value in categorical columns completed")

def FillDoctorInCharge():
    df["DoctorInCharge"] = df["DoctorInCharge"].fillna("Dr_Confid")
    print("Process 4 : Manage DoctorInCharge columns completed")
    

def Main():
    Overview()
    
    print("Data Cleansing Process started :)")
    fillID()
    fillMed()
    fillMode()
    FillDoctorInCharge()
    
    print(df.head(10))
    df.to_csv("INT243_cleaned.csv", index=False)

Main()