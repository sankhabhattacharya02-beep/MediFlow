# MediFlow - Hospital Wait Time Predictor
# Step 1: Load and inspect the dataset

# Import the pandas library, which helps us work with tabular data (like spreadsheets)
import pandas as pd
import joblib

# Load the CSV file into a DataFrame (a table we can work with in Python)
df = pd.read_csv("Hospital_Wait_500.csv")

# Print how many rows and columns the dataset has
print("Number of rows:   ", df.shape[0])
print("Number of columns:", df.shape[1])

# Print a blank line for readability
print()

# Print the first 5 rows so we can see what the data looks like
print("First 5 rows of the dataset:")
print(df.head())

# ------------------------------------------------------------------
# Step 2: Select input features (X) and the target column (y)
# ------------------------------------------------------------------

# These are the 5 input columns the model will use to make predictions.
# All 5 are known at the moment triage is completed.
feature_columns = [
    "TriageCategory",           # Urgency level assigned by the triage nurse
    "Department",               # Clinical department the patient is in
    "FacilityOccupancyRate",    # How full the hospital is (0 to 1)
    "ProvidersOnShift",         # Number of doctors currently on duty
    "NurseToTriageCompleteTime" # Minutes from first nurse contact to triage done
]

# X holds the input features - the information we feed into the model
X = df[feature_columns]

# y holds the target - the value we want the model to predict (in minutes)
y = df["TriageToProviderStartTime"]

# Print the first 5 rows of the input features
print("Input features X (first 5 rows):")
print(X.head())

# Print a blank line for readability
print()

# Print the first 5 values of the target column
print("Target column y (first 5 values):")
print(y.head())

print("\n Input column data types: ")
print(X.dtypes)
print("\n Target data type: ",y.dtype)

# Convert text columns into numeric columns
X = pd.get_dummies(X, columns=["TriageCategory", "Department"], dtype=int)

# Check the converted input data
print("\nEncoded input features:")
print(X.head())
print("\nNumber of input columns after encoding:", X.shape[1])

# Save the encoded column names so Dashboard.py can reconstruct the same layout
joblib.dump(list(X.columns), "model_feature_columns.pkl")
print("\nFeature column names saved to model_feature_columns.pkl")

from sklearn.model_selection import train_test_split

# Keep 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

from sklearn.ensemble import RandomForestRegressor

# Create the prediction model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

print("\nModel training completed!")
# Predict waiting times for the test data
y_pred = model.predict(X_test)

# Compare actual and predicted waiting times
print("\nActual vs Predicted waiting time (minutes):")

for actual, predicted in zip(y_test.iloc[:5], y_pred[:5]):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f} minutes")

# Simple baseline: predict the same average time for everyone
average_wait = y_train.mean()

baseline_predictions = [average_wait] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print(f"\nBaseline MAE: {baseline_mae:.2f} minutes")
print(f"Random Forest MAE: {mae:.2f} minutes")

# ------------------------------------------------------------------
# Step 6: Save the trained model to a file
# ------------------------------------------------------------------

# joblib.dump() serialises the trained model object and writes it to disk.
# "wait_time_model.pkl" is the output file — .pkl is a standard binary format.
joblib.dump(model, "wait_time_model.pkl")

# Confirm the file was saved successfully
print("\nModel saved to wait_time_model.pkl")
