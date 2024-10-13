import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import numpy as np

# Load models
binary_model = XGBClassifier()
binary_model.load_model("binary_model.json")

multiclass_model = XGBClassifier()
multiclass_model.load_model("xgb_model_filtered.json")

# Streamlit App
st.title("Heartbeat Classification App")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Preprocess the uploaded file
    data = pd.read_csv(uploaded_file)

    # Check if the CSV file has the expected number of features
    st.write(f"Uploaded file has {data.shape[1]} features.")

    # Step 3: Binary classification - normal (0) or abnormal (1)
    st.write("Classifying each sample as normal or abnormal...")

    # Create a list to hold results
    results = []

    # Loop through each sample in the dataset
    for idx, row in data.iterrows():
        # Binary classification for normal/abnormal

        binary_prediction = binary_model.predict(np.array(row[:-1]).reshape(1, -1))

        if binary_prediction[0] == 0:
            results.append(
                {
                    "Sample": idx + 1,
                    "Binary Prediction": "Normal (0)",
                    "Multi-class Prediction": "-",
                }
            )
        else:
            # Multi-class classification if abnormal
            multiclass_prediction = multiclass_model.predict(
                np.array(row[:-1]).reshape(1, -1)
            )
            prediction_map = {
                0: "🟥 Supraventricular Ectopy Beats (S)",
                1: "🟨 Ventricular Ectopy Beats (V)",
                2: "🟦 Fusion Beats (F)",
                3: "🟪 Unclassifiable Beats (Q)",
            }
            results.append(
                {
                    "Sample": idx + 1,
                    "Binary Prediction": "Abnormal (1)",
                    "Multi-class Prediction": prediction_map[multiclass_prediction[0]],
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display the table
    st.write("Classification Results:")
    st.dataframe(results_df)
