import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_raw = pd.read_csv("asthma_disease_data_new.csv")

def fillID(data):
    start_id = 5034
    new_id = np.arange(start_id, start_id + len(data))
    data["PatientID"] = new_id
    print("Process 1 : Filling PatientID completed")
    return data


def fillMed(data):
    num_cols = ["Age","BMI","PhysicalActivity","DietQuality","SleepQuality","PollutionExposure","PollenExposure","DustExposure","LungFunctionFEV1","LungFunctionFVC"]
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
    print("Process 2 : Fill median values completed")
    return data

def fillMode(data):
    catego_cols = [
        "Gender",
        "Smoking",
        "PetAllergy",
        "FamilyHistoryAsthma",
        "HistoryOfAllergies",
        "Eczema",
        "HayFever",
        "GastroesophagealReflux",
        "Wheezing",
        "ShortnessOfBreath",
        "ChestTightness",
        "Coughing",
        "NighttimeSymptoms",
        "ExerciseInduced",
        "Diagnosis",
        "EducationLevel",
        "Ethnicity"
    ]
    for col in catego_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    print("Process 3 : Fill mode values completed")
    return data


def FillDoctorInCharge(data):
    data["DoctorInCharge"] = data["DoctorInCharge"].fillna("Dr_Confid")
    print("Process 4 : Fill DoctorInCharge completed")
    return data

def Preprocessing(data):
    data = fillID(data)
    data = fillMed(data)
    data = fillMode(data)
    data = FillDoctorInCharge(data)

    return data

def CompareMissing(raw_data, processed_data):
    raw_missing = raw_data.isna().sum()
    processed_missing = processed_data.isna().sum()
    compare = pd.DataFrame({
        "Raw Data": raw_missing,
        "Processed Data": processed_missing
    })
    print(compare)
    raw_incomplete = len(raw_data[raw_data.isna().any(axis=1)])
    processed_incomplete = len(processed_data[processed_data.isna().any(axis=1)])
    print(f"\nRaw incomplete records: {raw_incomplete}")
    print(f"Processed incomplete records: {processed_incomplete}")


def DemographicAnalysis(data):
    print("\nAge Statistics:")
    print(data["Age"].describe())

    print("\nGender Distribution:")
    print(data["Gender"].value_counts())

    plt.figure(figsize=(8,5))
    plt.hist(data["Age"], bins=10)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution")
    plt.show()

def FeatureAnalysis(data):
    print("\nFeature Analysis")
    feature_cols = [
        "BMI",
        "PhysicalActivity",
        "PollutionExposure",
        "PollenExposure",
        "DustExposure",
    ]
    print(data[feature_cols].describe())
    for col in feature_cols:
        plt.figure(figsize=(6,4))
        y_jitter = data["Diagnosis"] + np.random.normal(0,0.03,len(data))
        colors = data["Diagnosis"].map({0: "blue",1: "red"})
        plt.scatter(data[col],y_jitter,c=colors,s=10,alpha=0.5)
        plt.xlabel(col)
        plt.ylabel("Diagnosis")
        plt.title(f"{col} vs Asthma Diagnosis")
        plt.show()
        
    binary_cols = ["Smoking","Wheezing","ShortnessOfBreath","Coughing","NighttimeSymptoms"]
    print("\nBinary Features")
    for col in binary_cols:
        result = data.groupby([col, "Diagnosis"]).size().unstack(fill_value=0)

        print(f"\n{col}")
        print(result)

        result.plot(kind='bar',figsize=(7,5))
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"{col} vs no {col} that affected Asthma Diagnosis")
        plt.xticks([0,1],["No", "Yes"],rotation=0)
        plt.legend(["No Asthma","Asthma"])
        plt.show()

def DiagnosisCorrelation(data): 
    diagnosis_corr = df_raw.drop(columns=["PatientID"]).corr(numeric_only=True)[["Diagnosis"]].sort_values(by="Diagnosis", ascending=False)

    diagnosis_corr = diagnosis_corr.drop("Diagnosis")
    plt.figure(figsize=(6,10))
    plt.imshow(diagnosis_corr,aspect='auto')
    plt.colorbar()
    plt.yticks(range(len(diagnosis_corr.index)),diagnosis_corr.index)
    plt.xticks([0], ["Diagnosis"])
    plt.title("Correlation with Asthma Diagnosis")
    plt.show()

def main():
    raw_data = df_raw.copy()
    processed_data = Preprocessing(raw_data.copy())
    processed_data.to_csv("INT243_cleaned.csv",index=False)
    CompareMissing(raw_data,processed_data)
    DemographicAnalysis(processed_data)
    FeatureAnalysis(processed_data)
    DiagnosisCorrelation(processed_data)
    
main()