import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistem Informasi Reservasi & Rental Armada Kendaraan",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Sistem Informasi Reservasi & Rental Armada Kendaraan")
st.markdown("**Domain**: `Web & Sistem Informasi / Fleet Management` | **Tech Stack**: `React, Next.js, Modern JS, Tailwind Solid Dark`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    engine_rpm_mean = st.slider("Engine Rpm Mean", float(750.0), float(6500.0), float(2450.0))
    engine_coolant_temp_c = st.slider("Engine Coolant Temp C", float(65.0), float(125.0), float(92.0))
    oil_pressure_psi = st.slider("Oil Pressure Psi", float(10.0), float(75.0), float(42.0))
    hard_braking_events_per_100km = st.slider("Hard Braking Events Per 100Km", float(0.0), float(12.0), float(2.5))
    battery_voltage_v = st.slider("Battery Voltage V", float(10.5), float(15.2), float(13.8))
    odometer_reading_km_k = st.slider("Odometer Reading Km K", float(5.0), float(300.0), float(85.0))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"engine_rpm_mean": engine_rpm_mean, "engine_coolant_temp_c": engine_coolant_temp_c, "oil_pressure_psi": oil_pressure_psi, "hard_braking_events_per_100km": hard_braking_events_per_100km, "battery_voltage_v": battery_voltage_v, "odometer_reading_km_k": odometer_reading_km_k}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `component_failure_risk`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
