import numpy as np
import pandas as pd
import seaborn as sns #pip install seaborn <use for data visualization>
import matplotlib.pyplot as plt

df_raw = pd.read_csv('asthma_disease_data_new.csv')    

def clean_data(data):

    df = data.copy()

    numerical_columns = [
        "Age",
        "BMI",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "PollutionExposure",
        "PollenExposure",
        "DustExposure",
        "LungFunctionFEV1",
        "LungFunctionFVC"
    ]    

    categorical_columns = [
        "Gender",
        "Ethnicity"
    ]

    ordinal_columns = [
        "EducationLevel"
    ]

    binary_columns = [
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
    ]

    df["PatientID"] = df["PatientID"].interpolate(method= "linear").astype(int) #clean PID (fill up the missing value)
       
    for col in numerical_columns:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(np.nanmedian(df[col].values))

    for col in categorical_columns:
        df[col] = df[col].fillna(-1)

    for col in ordinal_columns:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(-1)

    for col in binary_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df



def compare_before_after(raw, cleaned, column): # for numerical data
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    sns.histplot(raw[column], kde=True, ax=axes[0])
    axes[0].set_title(f"{column} Before Cleaning")

    sns.histplot(cleaned[column], kde=True, ax=axes[1])
    axes[1].set_title(f"{column} After Cleaning")

    changed = df_raw[column] != cleaned[column]

    plt.tight_layout()
    plt.show()
    print(f"{column} Column total changed rows:", changed.sum())

    return changed

def main():
    df_clean = clean_data(df_raw.copy())
    compare_before_after(df_raw, df_clean, "SleepQuality")


if __name__ == "__main__":
    main()