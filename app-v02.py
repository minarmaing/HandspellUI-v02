import streamlit as st
import cv2
import mediapipe as mp
import requests
import numpy as np
import tempfile
import os

# 1. API Configuration - Points to your live V3 GCP Endpoint
API_URL = "https://handspell-landmarks-asl-api-v03-77400974703.europe-west1.run.app/predict"
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
    """
    Extracts, formats, and normalizes landmarks for BOTH left and right hands
    to match the 126-feature dimension required by the trained LSTM model.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Structure empty templates for both hands (21 nodes * 3 axes = 63 values each)
    left_hand_features = [0.0] * 63
    right_hand_features = [0.0] * 63

    # Check if any hands were detected
    if results.multi_hand_landmarks and results.multi_handedness:
        # Loop through detected hands and identify if they are Left or Right
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label # "Left" or "Right"

            raw_pts = []
            for lm in hand_landmarks.landmark:
                raw_pts.extend([lm.x, lm.y, lm.z])

            # Normalize coordinates relative to the wrist root point (the first item)
            base_x, base_y, base_z = raw_pts[0], raw_pts[1], raw_pts[2]
            norm_pts = []
            for i in range(0, len(raw_pts), 3):
                norm_pts.extend([raw_pts[i] - base_x, raw_pts[i+1] - base_y, raw_pts[i+2] - base_z])

            # Assign coordinates to the correct hand slot
            if len(norm_pts) == 63:
                if hand_type == "Left":
                    left_hand_features = norm_pts
                elif hand_type == "Right":
                    right_hand_features = norm_pts

    # Combine both hand features together to create the exact 126-feature row expected by the model
    # Layout order: Left Hand (63) + Right Hand (63) = 126
    # Swap the order if your training columns put Right hand features first:
    combined_features = right_hand_features + left_hand_features
    return combined_features


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

        # --- UPDATE THIS BLOCK INSIDE APP-V02.PY ---
        num_extracted_frames = len(video_frames_features)

        if num_extracted_frames > 1:
            st.info(f"Extracted {num_extracted_frames} frames. Interpolating sequence to 30 timesteps...")

            # 1. Generate 30 evenly distributed floating-point indices across your 10 frames
            # e.g., [0.0, 0.31, 0.62, ..., 9.0]
            target_indices = np.linspace(0, num_extracted_frames - 1, SEQUENCE_LENGTH)

            final_sequence = []
            for idx in target_indices:
                low_idx = int(np.floor(idx))
                high_idx = int(np.ceil(idx))
                weight = idx - low_idx

                # Grab the bounding frames
                frame_low = video_frames_features[low_idx]["features"]
                frame_high = video_frames_features[high_idx]["features"]

                # 2. Perform linear math blending to create smooth intermediate coordinates
                blended_features = []
                for f_low, f_high in zip(frame_low, frame_high):
                    blended_val = (1 - weight) * f_low + weight * f_high
                    blended_features.append(blended_val)

                final_sequence.append({"features": blended_features})

        elif num_extracted_frames == 1:
            # Fallback if the video only contains 1 single valid frame
            st.warning("Video too short. Duplicating single frame across timeline.")
            final_sequence = [video_frames_features[0]] * SEQUENCE_LENGTH
        else:
            # Fallback placeholder if no frames processed at all
            st.error("No hands detected anywhere in this video file.")
            empty_frame = {"features": [0.0] * 126}
            final_sequence = [empty_frame] * SEQUENCE_LENGTH

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
