import numpy as np
import pandas as pd


df = pd.read_csv('asthma_disease_data_new.csv')    
pd.set_option('display.max_rows',None)

def cleaning():
    
    PID = df["PatientID"]
    age = df["Age"]
    sex = df["Gender"]
    eth = df["Ethnicity"] 
    edu = df["EducationLevel"]
    phy = df["Physicalactivity"]
    diet = df["DietQuality"]
    sleep = df["SleepQuality"]
    pollution = df["PollutionExposure"]
    pollen = df["PollenExposure"]
    dust = df["DustExposure"]
    pet = df["PetAllergy"]
    family = df["FamilyHistoryAsthma"]
    history = df["HistoryOfAllergies"]
    eczema = df["Eczema"]
    hayfever = df["HayFever"]
    gastro = df["GastroesophagealReflux"]
    fev1 = df["LungFunctionFEV1"]
    fvc = df["LungFunctionFVC"]
    
    other_ethnicity = 3

    df["PatientID"] = df["PatientID"].interpolate(method= "linear").astype(int) #clean PID (fill up the missing value)
    df["Age"] = df["Age"].fillna(df["Age"].mean().round()) # fill NaN val with mean of age
    df["Ethnicity"] = df["Ethnicity"].fillna(other_ethnicity) # fill NaN val with 3(other ethnicity)


    # print(edu.value_counts(normalize=True)*100)
    # print(cleaned_eth)
    # print(cleaned_eth.value_counts(normalize=True)*100)
    # print(sex)

    return df

def clean_data(df):
    
    other_ethnicity = 3

    df["PatientID"] = df["PatientID"].interpolate(method= "linear").astype(int) #clean PID (fill up the missing value)
    
    df["Age_missing"] = df["Age"].isnull().astype(int) # if Age was missing Age_missing value => 1 if not => 0
    df["Age"] = df["Age"].fillna(df["Age"].median()) # fill NaN val with median of age
    
    df["Ethnicity"] = df["Ethnicity"].fillna(other_ethnicity) # fill NaN val with 3(other ethnicity)
    
    df["EducationLevel_missing"] = df["EducationLevel"].isnull().astype(int)
    df["EducationLevel"] = df["EducationLevel"].fillna("unknown") # nominal

    df["BMI_missing"] = df["BMI"].isnull().astype(int)
    df["BMI"] = df["BMI"].fillna(df["BMI"].median) # continuous

    df["Smoking_missing"] = df["Smoking"].isnull().astype(int)
    df["Smoking"] = df["Smoking"].fillna("unknown") # nominal

    
    return df

clean_data(df)