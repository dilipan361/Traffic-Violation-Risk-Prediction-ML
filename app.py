from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and encoder
model = joblib.load("traffic_violation_model.pkl")
le = joblib.load("label_encoder.pkl")

risk_map = {0: "Low", 1: "Medium", 2: "High"}

@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Traffic Violation Risk Prediction</title>
        <style>
            body { font-family: Arial; background: #f2f2f2; }
            .box { background: white; padding: 20px; width: 400px; margin: 50px auto; border-radius: 10px; }
            input, select { width: 100%; padding: 8px; margin: 5px 0; }
            button { width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>Traffic Violation Risk Prediction</h2>
            <form method="post" action="/predict">
                Age: <input type="number" name="age" required><br>
                
                Gender:
                <select name="gender">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>

                Vehicle:
                <select name="vehicle">
                    <option value="Car">Car</option>
                    <option value="Bike">Bike</option>
                    <option value="Bus">Bus</option>
                    <option value="Truck">Truck</option>
                </select>

                Speed (kmph): <input type="number" name="speed" required>

                Alcohol:
                <select name="alcohol">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>

                Seatbelt:
                <select name="seatbelt">
                    <option value="1">Used</option>
                    <option value="0">Not Used</option>
                </select>

                Signal Jumped:
                <select name="signal">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>

                Previous Violations: <input type="number" name="prev" required>

                Weather:
                <select name="weather">
                    <option value="Clear">Clear</option>
                    <option value="Rainy">Rainy</option>
                    <option value="Foggy">Foggy</option>
                </select>

                Time of Day:
                <select name="time">
                    <option value="Morning">Morning</option>
                    <option value="Afternoon">Afternoon</option>
                    <option value="Evening">Evening</option>
                    <option value="Night">Night</option>
                </select>

                <br><br>
                <button type="submit">Predict Risk</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = request.form["gender"]
    vehicle = request.form["vehicle"]
    speed = int(request.form["speed"])
    alcohol = int(request.form["alcohol"])
    seatbelt = int(request.form["seatbelt"])
    signal = int(request.form["signal"])
    prev = int(request.form["prev"])
    weather = request.form["weather"]
    time = request.form["time"]

    # Encode categorical values (same order used in training)
    gender = le.fit_transform([gender])[0]
    vehicle = le.fit_transform([vehicle])[0]
    weather = le.fit_transform([weather])[0]
    time = le.fit_transform([time])[0]

    input_data = np.array([[age, gender, vehicle, speed, alcohol, seatbelt, signal, prev, weather, time]])

    prediction = model.predict(input_data)
    result = risk_map[prediction[0]]

    return f"""
    <h2 style='text-align:center;'>Predicted Traffic Violation Risk: {result}</h2>
    <center><a href="/">Go Back</a></center>
    """

if __name__ == "__main__":
    app.run(debug=True)
