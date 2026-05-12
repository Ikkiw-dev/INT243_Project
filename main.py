import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("asthma_disease_data_new.csv")

def fillID():
    start_id = 5034
    new_id = np.arange(start_id, start_id + len(df))
    df["PatientID"] = new_id

def fillMed():
    num_cols = ["Age", "BMI", "PhysicalActivity", "DietQuality",
            "SleepQuality", "PollutionExposure", "PollenExposure",
            "DustExposure", "LungFunctionFEV1", "LungFunctionFVC"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

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
    
def Preprocessing():
    fillID()
    fillMed()
    fillMode()
    FillDoctorInCharge()
    
    print(df.head(10))
    df.to_csv("INT243_cleaned.csv", index=False)

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

def DiagnosisCorrelation():
    print("Diagnosis Correlation")
    corr_df = df.drop(columns=["PatientID"])
    correlation = corr_df.corr(numeric_only=True)
    diagnosis_corr = correlation[["Diagnosis"]].sort_values(by = "Diagnosis",ascending=False)
    print(diagnosis_corr)
    diagnosis_corr = diagnosis_corr.drop("Diagnosis")
    plt.figure(figsize=(6,10))
    plt.imshow(diagnosis_corr,aspect='auto')
    plt.colorbar()
    plt.yticks(range(len(diagnosis_corr.index)),diagnosis_corr.index)
    plt.xticks([0], ["Diagnosis"])
    plt.title("Correlation with Asthma Diagnosis")
    plt.show()

def main():
    Preprocessing()
    demoanalyze()
    LifestyleAnalysis()
    EnvironmentalAnalysis()
    MedAnalysis()
    DiagnosisCorrelation()

main()