import pandas as pd
import numpy as np
from datetime import datetime

# 1. EXTRACT: Load the raw data of the ED visits
print("--- Starting ETL Process ---")
df = pd.read_csv('CN_RAW_DATA.csv')
initial_row_count = len(df)

# 2. TRANSFORM: Cleaning and standardization

# 2.1 Remove duplicates
df = df.drop_duplicates()
print(f"Removed {initial_row_count - len(df)} duplicate rows.")

# 2.2 Standardize timestamps
df['Time_Arrival'] = pd.to_datetime(df['Time_Arrival'], errors='coerce')
df['Time_Triage'] = pd.to_datetime(df['Time_Triage'], errors='coerce')
df['Time_Doc_Seen'] = pd.to_datetime(df['Time_Doc_Seen'], errors='coerce')

# 2.3 Handle missing arrival times
mask_impute = df['Time_Arrival'].isnull() & df['Time_Triage'].notnull()
df.loc[mask_impute, 'Time_Arrival'] = df.loc[mask_impute, 'Time_Triage'] - pd.Timedelta(minutes=15)

# Drop rows where both arrival and triage are missing
df = df.dropna(subset=['Time_Arrival', 'Time_Triage'], how='all')

# 2.4 Fix logic errors (Triage before arrival)
mask_logic = df['Time_Triage'] < df['Time_Arrival']
df.loc[mask_logic, 'Time_Triage'] = df.loc[mask_logic, 'Time_Arrival'] + pd.Timedelta(minutes=5)

## 2.5 Clean and Filter Gender Column
# 1. Standardize text (remove spaces and lowercase)
df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()

# 2. Define the valid mapping
gender_map = {
    'm': 'Male', 
    'male': 'Male',
    'f': 'Female', 
    'female': 'Female'
}
df['Gender'] = df['Gender'].map(gender_map) # Map to standard values, others become NaN(not a number)

# 3. REMOVE rows where Gender is now NaN (Unknown/Other)
initial_count_before_gender = len(df)
df = df.dropna(subset=['Gender'])
removed_gender_count = initial_count_before_gender - len(df)

print(f"Removed {removed_gender_count} rows with 'Unknown' or 'Other' gender.")

# 2.6 Clean age column 
df = df[(df['Age'] >= 0) & (df['Age'] <= 110)]

# 2.7 Handle missing ESI acuity and foreign keys
# Fill missing ESI with the median (3) to avoid breaking dashboard filters
df['ESI_Acuity'] = df['ESI_Acuity'].fillna(3).astype(int)
df['FK_Patient_ID'] = df['FK_Patient_ID'].fillna(-1).astype(int)

# 2.8 Fix boarding time outliers
# Cap boarding time at 24 hours (1440 mins)
df.loc[df['Boarding_Time_min'] > 1440, 'Boarding_Time_min'] = 1440

## 2.9 Calculate LWBS_Flag
# A patient is LWBS if they have an Arrival time but NO "Time_Doc_Seen"
df['LWBS_Flag'] = 0 
df.loc[df['Time_Doc_Seen'].isnull(), 'LWBS_Flag'] = 1

print(f"Identified {df['LWBS_Flag'].sum()} LWBS cases during ETL.")

# 3. LOAD: Save the clean dataset
clean_filename = 'CN_CLEAN_DATA.csv'
try:
    df.to_csv(clean_filename, index=False)
    saved_name = clean_filename
except PermissionError:
    saved_name = f"CN_CLEAN_DATA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(saved_name, index=False)

print(f"--- ETL Complete ---")
print(f"Final Row Count: {len(df)}")
print(f"Cleaned data saved to: {saved_name}")