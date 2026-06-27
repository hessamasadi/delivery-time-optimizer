import streamlit as st
import pandas as pd
import pickle
import numpy as np

@st.cache_resource
def load_artifacts():
    with open(r'C:\Users\HessaM\Desktop\dataset\archive\preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open(r'C:\Users\HessaM\Desktop\dataset\archive\model.pkl', 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_artifacts()

st.set_page_config(page_title="Delivery Time Predictor", layout="centered")

st.title("Delivery Time Predictor & Vehicle Optimizer")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        distance = st.number_input("Distance (km)", min_value=0.0, max_value=25.0, value=10.0, step=0.5)
        weather = st.selectbox("Weather", ["Clear", "Rainy", "Snowy", "Windy", "Foggy"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    
    with col2:
        prep_time = st.number_input("Preparation Time (min)", min_value=0, max_value=30, value=15, step=1)
        experience = st.number_input("Courier Experience (years)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        vehicle = st.selectbox("Vehicle Type (for prediction)", ["Bike", "Car", "Scooter"])
    
    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:
    input_data = pd.DataFrame([{
        'Distance_km': distance,
        'Weather': weather,
        'Traffic_Level': traffic,
        'Time_of_Day': time_of_day,
        'Vehicle_Type': vehicle,
        'Preparation_Time_min': prep_time,
        'Courier_Experience_yrs': experience,
        'Weather_Unknown': 0,
        'Traffic_Unknown': 0,
        'Time_Unknown': 0,
        'Experience_Unknown': 0
    }])
    
    input_enc = preprocessor.transform(input_data)
    pred = model.predict(input_enc)[0]
    st.success(f"Predicted Delivery Time: **{pred:.1f} minutes**")

st.divider()
st.subheader("Optimize Vehicle Selection")

with st.form("optimize_form"):
    col1, col2 = st.columns(2)
    with col1:
        opt_distance = st.number_input("Distance (km) for optimization", min_value=0.0, max_value=25.0, value=10.0, step=0.5, key="opt_dist")
        opt_weather = st.selectbox("Weather for optimization", ["Clear", "Rainy", "Snowy", "Windy", "Foggy"], key="opt_weather")
        opt_traffic = st.selectbox("Traffic Level for optimization", ["Low", "Medium", "High"], key="opt_traffic")
    with col2:
        opt_time = st.selectbox("Time of Day for optimization", ["Morning", "Afternoon", "Evening", "Night"], key="opt_time")
        opt_prep = st.number_input("Preparation Time (min) for optimization", min_value=0, max_value=30, value=15, step=1, key="opt_prep")
        opt_exp = st.number_input("Courier Experience (years) for optimization", min_value=0.0, max_value=10.0, value=5.0, step=0.5, key="opt_exp")
    
    optimize_clicked = st.form_submit_button("Find Best Vehicle")

if optimize_clicked:
    base = {
        'Distance_km': opt_distance,
        'Weather': opt_weather,
        'Traffic_Level': opt_traffic,
        'Time_of_Day': opt_time,
        'Preparation_Time_min': opt_prep,
        'Courier_Experience_yrs': opt_exp,
        'Weather_Unknown': 0,
        'Traffic_Unknown': 0,
        'Time_Unknown': 0,
        'Experience_Unknown': 0
    }
    
    results = {}
    for v in ['Bike', 'Car', 'Scooter']:
        row = base.copy()
        row['Vehicle_Type'] = v
        df_row = pd.DataFrame([row])
        enc = preprocessor.transform(df_row)
        pred = model.predict(enc)[0]
        results[v] = pred
    
    best = min(results, key=results.get)
    
    st.write("Predicted delivery times:")
    for v, t in results.items():
        st.write(f"  - {v}: **{t:.1f}** minutes")
    
    st.success(f"Optimal Vehicle: **{best}** ({results[best]:.1f} minutes)")

st.caption("Model: Linear Regression | MAE: 5.92 min | R²: 0.829")