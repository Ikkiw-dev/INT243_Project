import numpy as np
import pandas as pd

df = pd.read_csv('asthma_disease_data_new.csv')    
pd.set_option('display.max_rows',None)

def cleaning():
    #clean PID (fill up the missing value)
    PID = df["PatientID"]
    cleaned_PID = PID.interpolate(method= "linear").astype(int)
    print(cleaned_PID)

cleaning()