from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load Trained Model
model = joblib.load("Models/random_forest_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get User Input
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    occupation = int(request.form["occupation"])
    screen_time = float(request.form["screen_time"])
    sleep = float(request.form["sleep"])
    activity = int(request.form["activity"])
    notifications = int(request.form["notifications"])

    # Create DataFrame for Prediction
    sample = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "occupation": [occupation],
        "daily_screen_hours": [screen_time],
        "avg_sleep_hours": [sleep],
        "physical_activity_days_per_week": [activity],
        "daily_notifications": [notifications]
    })

    # Prediction
    pred = model.predict(sample)[0]

    # Convert Numeric Prediction to Label
    labels = {
        0: "At-risk",
        1: "Good",
        2: "Moderate"
    }

    prediction = labels[pred]

    # Wellness Score
    score = 100

    if screen_time > 8:
        score -= 25
    elif screen_time > 6:
        score -= 15

    if sleep < 6:
        score -= 20
    elif sleep < 7:
        score -= 10

    if activity < 2:
        score -= 15

    if notifications > 150:
        score -= 15

    if score < 0:
        score = 0

    # Wellness Level
    if score >= 80:
        level = "Excellent 😊"
    elif score >= 60:
        level = "Moderate 🙂"
    else:
        level = "At Risk ⚠"

    # Recommendations
    recommendations = []

    if screen_time > 6:
        recommendations.append("Reduce your daily screen time.")

    if sleep < 7:
        recommendations.append("Sleep at least 7-8 hours daily.")

    if activity < 3:
        recommendations.append("Increase physical activity.")

    if notifications > 100:
        recommendations.append("Turn off unnecessary notifications.")

    return render_template(
        "result.html",
        prediction=prediction,
        score=score,
        level=level,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)