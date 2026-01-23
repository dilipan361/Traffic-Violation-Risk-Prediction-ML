import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
st.title("🚦 Traffic Violation Risk Prediction")
data = pd.read_csv("traffic_violation_dataset.csv")
le_density = LabelEncoder()
le_risk = LabelEncoder()
data["traffic_density"] = le_density.fit_transform(data["traffic_density"])
data["risk_level"] = le_risk.fit_transform(data["risk_level"])
X = data[["speed", "signal_jump", "helmet_seatbelt", "past_violations", "traffic_density"]]
y = data["risk_level"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
st.subheader("Enter Traffic Details")
speed = st.number_input("Speed (km/h)", min_value=0, max_value=200, value=50)
signal_jump = st.selectbox("Signal Jump", ["No", "Yes"])
helmet_seatbelt = st.selectbox("Helmet / Seatbelt Worn", ["Yes", "No"])
past_violations = st.number_input("Past Violations", min_value=0, max_value=20, value=0)
traffic_density = st.selectbox("Traffic Density", ["Low", "Medium", "High"])
signal_jump = 1 if signal_jump == "Yes" else 0
helmet_seatbelt = 1 if helmet_seatbelt == "Yes" else 0
traffic_density = le_density.transform([traffic_density])[0]
if st.button("Predict Risk"):
    user_data = [[speed, signal_jump, helmet_seatbelt, past_violations, traffic_density]]
    prediction = model.predict(user_data)
    risk = le_risk.inverse_transform(prediction)[0]
    st.markdown("### 🧾 Prediction Result")
    st.write(f"**Predicted Traffic Violation Risk Level:** {risk}")
    if risk == "High":
        st.error("🚨 High risk of traffic violation!")
    elif risk == "Medium":
        st.warning("⚠️ Medium risk of traffic violation.")
    else:
        st.success("✅ Low risk of traffic violation.")
