import streamlit as st
import cv2
import mediapipe as mp
import requests
import numpy as np
import tempfile
import os

# 1. API Configuration - Points to your live V3 GCP Endpoint
API_URL = "https://handspell-landmarks-asl-api-v03-77400974703.europe-west1.run.app"
SEQUENCE_LENGTH = 30

# 2. Initialize MediaPipe Hands Solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 3. Streamlit Page Setup UI
st.set_page_config(page_title="Team Handspell V3", page_icon="✋", layout="centered")
st.title("✋ ASL Sequence Recognition (V3 LSTM)")
st.write("Evaluate dynamic sign language gestures using our cloud-hosted recurrent neural network.")

# Interactive Mode Selection Toggle
mode = st.radio("Choose Input Testing Source:", ["Live Web Camera", "Upload Video File (.mp4)"])

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

def process_landmarks_from_frame(frame):
    """Helper function to extract, format, and normalize landmarks from a single frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_frame_features = [0.0] * 63

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            raw_pts = []
            for lm in hand_landmarks.landmark:
                raw_pts.extend([lm.x, lm.y, lm.z])

            # Center-align spatial metrics relative to the wrist root point
            base_x, base_y, base_z = raw_pts[0], raw_pts[1], raw_pts[2]
            norm_pts = []
            for i in range(0, len(raw_pts), 3):
                norm_pts.extend([raw_pts[i] - base_x, raw_pts[i+1] - base_y, raw_pts[i+2] - base_z])

            if len(norm_pts) == 63:
                current_frame_features = norm_pts
    return current_frame_features

# ==========================================
# MODE A: LIVE WEB CAMERA STREAMING
# ==========================================
if mode == "Live Web Camera":
    run_app = st.checkbox("Toggle Web Camera On/Off", value=False)
    frame_buffer = []  # Rolling timeline sliding window queue

    if run_app:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and run_app:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            features = process_landmarks_from_frame(frame)
            frame_buffer.append({"features": features})

            # Restrict queue size to exactly 30 frames
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            # Trigger cloud inference when sliding window buffer is full
            if len(frame_buffer) == SEQUENCE_LENGTH:
                try:
                    response = requests.post(API_URL, json={"sequence": frame_buffer}, timeout=1.5)
                    if response.status_code == 200:
                        data = response.json()
                        gesture = data.get("prediction", "Unknown")
                        conf = data.get("confidence_score", 0.0)
                        prediction_placeholder.markdown(f"## Gesture Detected: **{gesture}** *({conf*100:.1f}%)*")
                except Exception as err:
                    prediction_placeholder.warning(f"Connection timeout: {err}")

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

# ==========================================
# MODE B: UPLOAD VIDEO FILE (.MP4)
# ==========================================
else:
    uploaded_file = st.file_uploader("Select an ASL gesture video clip...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create a temporary local file to stream the uploaded binary clip data safely into OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        video_frames_features = []

        st.info("Processing uploaded video file frames...")

        # Read the entire video file sequentially to harvest all coordinate states
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            features = process_landmarks_from_frame(frame)
            video_frames_features.append({"features": features})

        cap.release()
        os.unlink(tfile.name)  # Clean up temp file storage disk space

        # --- Feature Alignment Boundary Rules ---
        # Reshape or pad the video sequence list to fit your LSTM input requirements (exactly 30 frames)
        if len(video_frames_features) >= SEQUENCE_LENGTH:
            # Truncate clips that are too long
            final_sequence = video_frames_features[:SEQUENCE_LENGTH]
        else:
            # Pad shorter video files with empty arrays to reach the 30-frame milestone
            padding_len = SEQUENCE_LENGTH - len(video_frames_features)
            empty_frame = {"features": [0.0] * 63}
            final_sequence = video_frames_features + [empty_frame] * padding_len

        # Send the complete, compiled 30-frame sequence to your live GCP API endpoint
        with st.spinner("Analyzing sequence transitions on GCP Cloud Run..."):
            try:
                payload = {"sequence": final_sequence}
                response = requests.post(API_URL, json=payload, timeout=3.0)

                if response.status_code == 200:
                    data = response.json()
                    gesture = data.get("prediction", "Unknown")
                    conf = data.get("confidence_score", 0.0)

                    prediction_placeholder.markdown(
                        f"🎉 **Evaluation Complete!**\n"
                        f"## Predicted Gesture: **{gesture}** *(Confidence: {conf*100:.1f}%)*"
                    )
                else:
                    st.error(f"GCP Server returned error code {response.status_code}: {response.text}")
            except Exception as err:
                st.error(f"Failed to communicate with production API endpoint: {err}")
