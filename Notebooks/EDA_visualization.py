import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Data/social_media_screentime_mental_health_2026.csv")

# Bar chart
df["wellbeing_band"].value_counts().plot(kind="bar")

plt.title("Wellbeing Band Distribution")
plt.xlabel("Wellbeing Band")
plt.ylabel("Number of Users")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df["daily_screen_hours"], bins=20)

plt.title("Daily Screen Time Distribution")
plt.xlabel("Daily Screen Hours")
plt.ylabel("Number of Users")

plt.show()

plt.figure(figsize=(6,4))

df["gender"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8,5))

df["most_used_platform"].value_counts().plot(kind="bar")

plt.title("Most Used Social Media Platform")
plt.xlabel("Platform")
plt.ylabel("Users")

plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(8,5))

df["occupation"].value_counts().plot(kind="bar")

plt.title("Occupation Distribution")
plt.xlabel("Occupation")
plt.ylabel("Users")

plt.xticks(rotation=45)

plt.show() 

plt.figure(figsize=(8,5))

plt.hist(df["anxiety_score_0to27"], bins=15)

plt.title("Anxiety Score Distribution")
plt.xlabel("Anxiety Score")
plt.ylabel("Users")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df["avg_sleep_hours"], bins=15)

plt.title("Average Sleep Hours")
plt.xlabel("Sleep Hours")
plt.ylabel("Users")

plt.show()

import seaborn as sns

plt.figure(figsize=(12,8))

sns.heatmap(df.select_dtypes(include=["int64","float64"]).corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

plt.figure(figsize=(8,5))

df.boxplot(column="daily_screen_hours", by="wellbeing_band")

plt.title("Screen Time vs Wellbeing")
plt.suptitle("")

plt.xlabel("Wellbeing Band")
plt.ylabel("Daily Screen Hours")

plt.show()