import argparse
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

DATA_PATH = "Hospital_Wait_500.csv"
MODEL_PATH = "wait_time_model.pkl"
FEATURES_PATH = "model_feature_columns.pkl"
FEATURE_COLUMNS = [
    "TriageCategory",
    "Department",
    "FacilityOccupancyRate",
    "ProvidersOnShift",
    "NurseToTriageCompleteTime",
]


def train_model(force_retrain: bool = False):
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["TriageToProviderStartTime"]

    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    X = pd.get_dummies(X, columns=["TriageCategory", "Department"], dtype=int)
    joblib.dump(list(X.columns), FEATURES_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    average_wait = y_train.mean()
    baseline_mae = mean_absolute_error(y_test, [average_wait] * len(y_test))

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("\nModel training completed!")
    print(f"\nMean Absolute Error: {mae:.2f} minutes")
    print(f"Baseline MAE: {baseline_mae:.2f} minutes")

    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to wait_time_model.pkl")
    print("Feature column names saved to model_feature_columns.pkl")

    return model, list(X.columns)


@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        model, _ = train_model(force_retrain=True)
        return model
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


@st.cache_resource
def load_feature_columns():
    try:
        if os.path.exists(FEATURES_PATH):
            return joblib.load(FEATURES_PATH)
        _, feature_columns = train_model(force_retrain=True)
        return feature_columns
    except Exception as exc:
        st.error(f"Failed to load feature columns: {exc}")
        return None


def predict_waiting_time(model, feature_columns, triage_category, department, facility_occupancy_rate,
                        providers_on_shift, nurse_to_triage_complete_time):
    input_df = pd.DataFrame([{
        "TriageCategory": triage_category,
        "Department": department,
        "FacilityOccupancyRate": facility_occupancy_rate,
        "ProvidersOnShift": providers_on_shift,
        "NurseToTriageCompleteTime": nurse_to_triage_complete_time,
    }])

    input_encoded = pd.get_dummies(input_df, columns=["TriageCategory", "Department"], dtype=int)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    return prediction


def dashboard():
    st.set_page_config(page_title="MediFlow", layout="centered")

    model = load_model()
    feature_columns = load_feature_columns()

    st.title("MediFlow — Smart Hospital Waiting Time Predictor")
    st.write("Enter patient details to estimate waiting time after triage.")

    triage_category = st.selectbox(
        "Triage Category",
        ["Emergency", "Immediate", "Non-urgent", "Semi-urgent", "Urgent"],
    )

    department = st.selectbox(
        "Department",
        [
            "Cardiology",
            "Emergency",
            "General Surgery",
            "Internal Medicine",
            "Neurology",
            "Obstetrics",
            "Oncology",
            "Orthopedics",
            "Pediatrics",
            "Radiology",
        ],
    )

    facility_occupancy_rate = st.number_input(
        "Facility Occupancy Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    providers_on_shift = st.number_input(
        "Providers on Shift",
        min_value=0,
        value=5,
        step=1,
    )

    nurse_to_triage_complete_time = st.number_input(
        "Nurse to Triage Complete Time (minutes)",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )

    if st.button("Predict Waiting Time"):
        if model is None or feature_columns is None:
            st.error("Model or feature columns not loaded. Cannot make a prediction.")
            return

        prediction = predict_waiting_time(
            model,
            feature_columns,
            triage_category,
            department,
            facility_occupancy_rate,
            providers_on_shift,
            nurse_to_triage_complete_time,
        )

        st.success(f"Estimated waiting time: **{prediction:.2f} minutes**")
        st.info(
            "This is an estimated waiting time for demonstration purposes, not a guaranteed time. "
            "Patients needing urgent care should follow hospital staff instructions."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediFlow hospital wait-time predictor")
    parser.add_argument("--train", action="store_true", help="Train the model and exit.")
    args = parser.parse_args()

    if args.train:
        train_model(force_retrain=True)
        raise SystemExit(0)

    if "streamlit" in sys.modules:
        dashboard()
    else:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
