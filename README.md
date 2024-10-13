# Heartbeat Classification App

This is a Streamlit app that classifies heartbeats based on an uploaded CSV file. It uses two models:
- A **binary classifier** to determine if a patient is normal (0) or abnormal (1).
- A **multi-class classifier** to identify the type of abnormal heartbeat.

## Features
- Upload a CSV file with 186 features per sample.
- Get a classification: normal or abnormal.
- If abnormal, the app will further classify the heartbeat as:
  - 🟥 **Supraventricular Ectopy Beats (S)**
  - 🟨 **Ventricular Ectopy Beats (V)**
  - 🟦 **Fusion Beats (F)**
  - 🟪 **Unclassifiable Beats (Q)**

## Requirements
- Python 3.x
- Streamlit
- XGBoost
- Pandas

