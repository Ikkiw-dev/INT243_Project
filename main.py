import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print("Process 4 : Manage DoctorInCharge columns completed\n")
    
def Preprocessing():
    Overview()
    
    print("\nData Cleansing Process started :)\n")
    fillID()
    fillMed()
    fillMode()
    FillDoctorInCharge()
    
    print(df.head(10))
    df.to_csv("INT243_cleaned.csv", index=False)

Preprocessing()

def validate():
    print("Data Validation Process")
    
    invalid_age = df[(df["Age"] < 5) | (df["Age"] > 80)]
    print(f"Invalid Age rows : {len(invalid_age)}")
    invalid_bmi = df[(df["BMI"] < 15) | (df["BMI"] > 40)]
    print(f"Invalide BMI rows : {len(invalid_bmi)}")
    duplicates = df.duplicated().sum()
    print(f"duplicates rows detected : {duplicates}")
    
    print("\nFinal Missing checker")
    print(df.isna().sum())

#demographic analysis to see that age - gender affected to asthma or not

def demoanalyze():
    print("\nDemoGraphic Analysis")
    print(df["Age"].describe())
    
    print("\nGender Distribution :")
    print(df["Gender"].value_counts())
    
    plt.figure(figsize=(8,5))
    plt.hist(df["Age"], bins=10)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution")
    plt.show()

def LifestyleAnalysis():
    print("\nLifestyle Analysis")
    print(df[["BMI", "PhysicalActivity", "Smoking"]].describe())
    plt.figure(figsize=(6,4))
    df.boxplot(column="BMI", by="Diagnosis")
    plt.title("BMI vs Asthma Diagnosis")
    plt.suptitle("")
    plt.show()

def EnvironmentalAnalysis():
    print("Environmental Analysis")
    en_cols = ["PollutionExposure","PollenExposure","DustExposure"]
    print(df[en_cols].describe())

def MedAnalysis():
    print ("Medical History and Symptons")
    sympton_cols = ["Wheezing","ShortnessOfBreath","Coughing","NighttimeSymptoms"]
    for col in sympton_cols :
        print(f"\n{col}:")
        print(df[col].value_counts())

def main():
    Preprocessing()
    validate()
    demoanalyze()
    LifestyleAnalysis()
    EnvironmentalAnalysis()
    MedAnalysis()

main()