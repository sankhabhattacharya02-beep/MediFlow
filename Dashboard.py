import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="MediFlow", layout="centered")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("wait_time_model.pkl")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_feature_columns():
    try:
        return joblib.load("model_feature_columns.pkl")
    except Exception as e:
        st.error(f"Failed to load feature columns: {e}")
        return None

model = load_model()
feature_columns = load_feature_columns()

st.title("MediFlow — Smart Hospital Waiting Time Predictor")
st.write("Enter patient details to estimate waiting time after triage.")

triage_category = st.selectbox(
    "Triage Category",
    ["Emergency", "Immediate", "Non-urgent", "Semi-urgent", "Urgent"]
)

department = st.selectbox(
    "Department",
    ["Cardiology", "Emergency", "General Surgery", "Internal Medicine",
     "Neurology", "Obstetrics", "Oncology", "Orthopedics", "Pediatrics", "Radiology"]
)

facility_occupancy_rate = st.number_input(
    "Facility Occupancy Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

providers_on_shift = st.number_input(
    "Providers on Shift",
    min_value=0,
    value=5,
    step=1
)

nurse_to_triage_complete_time = st.number_input(
    "Nurse to Triage Complete Time (minutes)",
    min_value=0.0,
    value=10.0,
    step=0.5
)

if st.button("Predict Waiting Time"):
    if model is None or feature_columns is None:
        st.error("Model or feature columns not loaded. Cannot make a prediction.")
    else:
        input_df = pd.DataFrame([{
            "TriageCategory":            triage_category,
            "Department":                department,
            "FacilityOccupancyRate":     facility_occupancy_rate,
            "ProvidersOnShift":          providers_on_shift,
            "NurseToTriageCompleteTime": nurse_to_triage_complete_time,
        }])

        input_encoded = pd.get_dummies(
            input_df,
            columns=["TriageCategory", "Department"],
            dtype=int
        )

        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        st.success(f"Estimated waiting time: **{prediction:.2f} minutes**")
        st.info(
            "This is an estimated waiting time for demonstration purposes, not a guaranteed time. "
            "Patients needing urgent care should follow hospital staff instructions."
        )
