import streamlit as st
import requests
import json
import os

# App configuration
st.set_page_config(page_title="ASL Gesture Recognition UI", layout="wide")

st.title("🖐️ Real-time & Video ASL Gesture Recognition")
st.write("Upload a short video clip (AVI/MP4/MOV) of an ASL sign to classify it using your cloud-deployed LSTM models.")

# Configuration Panel
st.sidebar.header("Configuration Panel")
model_type = st.sidebar.selectbox(
    "Choose Trained Model Context:",
    ["Wrist-Normalized Model", "Raw Coordinates Model"]
)

# API Endpoint mapping (Update /predict path depending on your exact FastAPI/Flask setup)
API_URL = "https://handspell-landmarks-asl-api-v03-77400974703.europe-west1.run.app/predict"

# Upload Section
uploaded_file = st.file_uploader("Choose a gesture video file...", type=["avi", "mp4", "mov"])

if uploaded_file is not None:
    # Display the uploaded video file directly in UI
    st.video(uploaded_file)

    if st.button("⚡ Run Cloud Inference"):
        with st.spinner("Streaming video to cloud API and processing gesture recognition..."):
            try:
                # Prepare payload file mapping
                # Reset file pointer to read from beginning
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"model_strategy": "normalized" if "Normalized" in model_type else "raw"}

                # Send HTTP POST request to your running API
                response = requests.post(API_URL, files=files, data=data, timeout=60)

                if response.status_code == 200:
                    result = response.json()

                    # Extract variables from API response schema
                    # Adjust dictionary keys to match your exact API JSON response structure
                    predicted_label = result.get("predicted_label", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    candidates = result.get("top_candidates", []) # Expected list of {"label": "A", "probability": 0.85}

                    # Visual results display
                    st.success(f"### Predicted Gesture: **{predicted_label}** (Confidence: {confidence:.2%})")

                    # Probability distribution visualizers
                    if candidates:
                        st.write("#### Top Gesture Candidates:")
                        for candidate in candidates:
                            lbl = candidate.get("label")
                            prob = candidate.get("probability", 0.0)
                            st.write(f"**{lbl}** ({prob:.2%})")
                            st.progress(float(prob))
                else:
                    st.error(f"API Error (Status Code {response.status_code}): {response.text}")

            except requests.exceptions.Timeout:
                st.error("The request timed out. Processing your video landmarks took too long on the server.")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the API. Verify that your cloud endpoint is up and running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
