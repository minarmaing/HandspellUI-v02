import streamlit as st
import cv2
import mediapipe as mp
import requests
import numpy as np

# 1. Point to your new live V3 LSTM Endpoint!
API_URL = "https://handspell-landmarks-asl-api-v03-77400974703.europe-west1.run.app/predict"
SEQUENCE_LENGTH = 30

# Initialize MediaPipe and Buffers
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
frame_buffer = []  # Holds the rolling timeline sequence of 30 frames

st.title("✋ Live ASL Sequence Recognition (V3 LSTM)")
run_app = st.checkbox("Toggle Web Camera", value=False)
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

if run_app:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and run_app:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Default empty frame array structure
        current_frame_features = [0.0] * 63

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                raw_pts = []
                for lm in hand_landmarks.landmark:
                    raw_pts.extend([lm.x, lm.y, lm.z])

                # Normalize relative to wrist root point
                base_x, base_y, base_z = raw_pts[0], raw_pts[1], raw_pts[2]
                norm_pts = []
                for i in range(0, len(raw_pts), 3):
                    norm_pts.extend([raw_pts[i] - base_x, raw_pts[i+1] - base_y, raw_pts[i+2] - base_z])

                if len(norm_pts) == 63:
                    current_frame_features = norm_pts

        # 2. Append the current frame coordinates to the timeline queue
        frame_buffer.append({"features": current_frame_features})

        # Maintain the sliding window constraints
        if len(frame_buffer) > SEQUENCE_LENGTH:
            frame_buffer.pop(0)

        # 3. Trigger cloud inference only when the sliding window has a full 1-second history
        if len(frame_buffer) == SEQUENCE_LENGTH:
            try:
                # Wrap the buffer inside the nested 'sequence' JSON key expected by your Pydantic schema
                payload = {"sequence": frame_buffer}
                response = requests.post(API_URL, json=payload, timeout=1.5)

                if response.status_code == 200:
                    data = response.json()
                    gesture = data.get("prediction", "Unknown")
                    confidence = data.get("confidence_score", 0.0)
                    prediction_placeholder.markdown(f"## Gesture Detected: **{gesture}** *({confidence*100:.1f}%)*")
            except Exception as err:
                prediction_placeholder.warning(f"Connection timeout: {err}")

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()
    hands.close()
