import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load Dataset
df = pd.read_csv("Data/social_media_screentime_mental_health_2026.csv")

# Fill Missing Values
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
df["avg_sleep_hours"] = df["avg_sleep_hours"].fillna(df["avg_sleep_hours"].mean())

print(df.head())

le = LabelEncoder()

categorical_columns = [
    "gender",
    "occupation",
    "region",
    "most_used_platform",
    "night_time_use",
    "primary_purpose",
    "uses_screen_time_limits",
    "attempted_digital_detox",
    "seeks_mental_health_support",
    "wellbeing_band"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print(df.head())

# Select only useful features
features = [
    "age",
    "gender",
    "occupation",
    "daily_screen_hours",
    "avg_sleep_hours",
    "physical_activity_days_per_week",
    "daily_notifications"
]

X = df[features]
y = df["wellbeing_band"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

########------Decision Tree Model------#######
# Create Model
dt_model = DecisionTreeClassifier(random_state=42)

# Train Model
dt_model.fit(X_train, y_train)

# Prediction
y_pred = dt_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

########------Random Forest Model------#######
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

#########--------Logistic Regression------#########
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(
    solver="saga",
    max_iter=10000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)

import matplotlib.pyplot as plt
import pandas as pd

importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print(importance)

plt.figure(figsize=(10,6))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

import joblib

joblib.dump(rf_model, "Models/random_forest_model.pkl")

print("Model Saved Successfully!")

########-------Model Evaluation------########
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, rf_pred))

######-----Feature Importance------######
import matplotlib.pyplot as plt

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(10,6))
plt.barh(feature_importance["Feature"],
         feature_importance["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy,
        lr_accuracy
    ]
})

print("\nModel Comparison:\n")
print(comparison)

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.show()

best_model = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("\nBest Model")
print(best_model)