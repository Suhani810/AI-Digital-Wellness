import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/social_media_screentime_mental_health_2026.csv")

print(df.head())

print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nInformation:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique Values in Target:")
print(df["wellbeing_band"].value_counts())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing Gender with Mode
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

# Fill missing Sleep Hours with Mean
df["avg_sleep_hours"] = df["avg_sleep_hours"].fillna(df["avg_sleep_hours"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())