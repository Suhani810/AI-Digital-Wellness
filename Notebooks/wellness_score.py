def calculate_score(screen_time, sleep, activity, notifications):

    score = 100

    # Screen Time
    if screen_time > 8:
        score -= 25
    elif screen_time > 6:
        score -= 15

    # Sleep
    if sleep < 6:
        score -= 20
    elif sleep < 7:
        score -= 10

    # Physical Activity
    if activity < 2:
        score -= 15

    # Notifications
    if notifications > 150:
        score -= 15

    if score < 0:
        score = 0

    return score

score = calculate_score(
    screen_time=8,
    sleep=6,
    activity=2,
    notifications=150
)

print("Wellness Score:", score)

#######---Smart Recommendation Engine---#####
recommendations = []

if 8 > 6:
    recommendations.append("Reduce your daily screen time.")

if 6 < 7:
    recommendations.append("Sleep at least 7-8 hours daily.")

if 2 < 3:
    recommendations.append("Increase physical activity.")

if 150 > 100:
    recommendations.append("Turn off unnecessary notifications.")

print("\nRecommendations:")

for rec in recommendations:
    print("-", rec)


if score >= 80:
    level = "Excellent 😊"
elif score >= 60:
    level = "Moderate 🙂"
else:
    level = "At Risk ⚠"

print("Wellness Level:", level)

import pandas as pd

report = pd.DataFrame({
    "Wellness Score":[score],
    "Wellness Level":[level]
})

report.to_csv("wellness_report.csv",index=False)

print("Report Saved Successfully!")

