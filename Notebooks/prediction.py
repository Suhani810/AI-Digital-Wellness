import joblib
import pandas as pd

# Load Model
model = joblib.load("Models/random_forest_model.pkl")

# Sample User Input (Same 15 features used during training)
sample = pd.DataFrame({
    "age": [22],
    "gender": [1],
    "occupation": [2],
    "region": [1],
    "most_used_platform": [3],
    "platforms_used_count": [4],
    "daily_screen_hours": [8],
    "daily_notifications": [150],
    "night_time_use": [1],
    "minutes_to_first_check_after_waking": [5],
    "primary_purpose": [2],
    "avg_sleep_hours": [6],
    "physical_activity_days_per_week": [2],
    "uses_screen_time_limits": [0],
    "attempted_digital_detox": [1]
})

prediction = model.predict(sample)

print("Prediction:", prediction)

result = prediction[0]

if result == 0:
    print("Predicted Wellbeing: At-risk")
elif result == 1:
    print("Predicted Wellbeing: Good")
else:
    print("Predicted Wellbeing: Moderate")

if result == 0:
    print("\nRecommendations:")
    print("- Reduce screen time.")
    print("- Sleep at least 7–8 hours.")
    print("- Try a digital detox.")
    print("- Increase physical activity.")

elif result == 1:
    print("\nRecommendations:")
    print("- Maintain your healthy routine.")
    print("- Continue regular exercise.")
    print("- Keep screen time balanced.")

else:
    print("\nRecommendations:")
    print("- Reduce night-time phone usage.")
    print("- Take regular breaks from social media.")
    print("- Improve your sleep schedule.")