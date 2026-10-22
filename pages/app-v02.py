import streamlit as st

st.set_page_config(page_title="Team Handspell V3", page_icon="✋", layout="centered")

import cv2
import mediapipe as mp
import requests
import numpy as np
import tempfile
import os

st.warning("Experimental prototype — accuracy is being improved.")

if st.button("← Back to HandSpell Studio"):
    st.switch_page("app-v06.py")
# 1. API Configuration - Points to your live V3 GCP Endpoint
API_URL = "https://handspell-landmarks-asl-api-v03-77400974703.europe-west1.run.app/predict"
SEQUENCE_LENGTH = 30

# 2. Initialize MediaPipe Hands Solutions (FIXED: max_num_hands=2 to capture both paths)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 3. Streamlit Page Setup UI

st.title("✋ ASL Sequence Recognition (V3 LSTM)")
st.write("Evaluate dynamic sign language gestures using our cloud-hosted recurrent neural network.")

# Interactive Mode Selection Toggle
mode = st.radio("Choose Input Testing Source:", ["Live Web Camera", "Upload Video File (.mp4)"])

frame_placeholder = st.empty()
prediction_placeholder = st.empty()
debug_placeholder = st.sidebar.empty() # Visual validation channel

def process_landmarks_from_frame(frame):
    """
    Extracts, formats, and normalizes landmarks for BOTH left and right hands
    to match the 126-feature dimension required by the trained LSTM model.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand_features = [0.0] * 63
    right_hand_features = [0.0] * 63
    hands_found_count = 0

    # Fixed extraction loops unpacking structure fields natively
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_found_count = len(results.multi_hand_landmarks)
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # FIXED: Bulletproof protobuf property accessor string extraction method
            hand_type = handedness.classification[0].label # "Left" or "Right"

            raw_pts = []
            for lm in hand_landmarks.landmark:
                raw_pts.extend([lm.x, lm.y, lm.z])

            # Use raw un-normalized coordinates to mirror your exact data pipeline
            if len(raw_pts) == 63:
                if hand_type == "Left":
                    left_hand_features = raw_pts
                elif hand_type == "Right":
                    right_hand_features = raw_pts

    # Combine both hand features together to form the 126 features vector layout
    # Layout matches Pandas alphabetical sort order: Left Hand (63) + Right Hand (63)
    combined_features = left_hand_features + right_hand_features
    return combined_features, hands_found_count


# ==========================================
# MODE A: LIVE WEB CAMERA STREAMING
# ==========================================
if mode == "Live Web Camera":
    run_app = st.checkbox("Toggle Web Camera On/Off", value=False)
    frame_buffer = []

    if run_app:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and run_app:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            features, _ = process_landmarks_from_frame(frame)
            frame_buffer.append({"features": features})

            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

# ==========================================
# MODE B: UPLOAD VIDEO FILE (.MP4)
# ==========================================
else:
    uploaded_file = st.file_uploader("Select an ASL gesture video clip...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        video_frames_features = []
        total_detections_logged = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            features, detected_count = process_landmarks_from_frame(frame)
            video_frames_features.append({"features": features})
            total_detections_logged += detected_count

        cap.release()
        os.unlink(tfile.name)

        num_extracted_frames = len(video_frames_features)

        # Display helpful pipeline monitoring telemetry onto the sidebar
        debug_placeholder.metric(
            label="MediaPipe Diagnostic Feed",
            value=f"{num_extracted_frames} Frames",
            delta=f"{total_detections_logged} Active Hand Samples"
        )

        if num_extracted_frames > 1:
            st.info(f"Extracted {num_extracted_frames} frames. Interpolating sequence to 30 timesteps...")
            target_indices = np.linspace(0, num_extracted_frames - 1, SEQUENCE_LENGTH)

            final_sequence = []
            for idx in target_indices:
                low_idx = int(np.floor(idx))
                high_idx = int(np.ceil(idx))
                weight = idx - low_idx

                frame_low = video_frames_features[low_idx]["features"]
                frame_high = video_frames_features[high_idx]["features"]

                blended_features = []
                for f_low, f_high in zip(frame_low, frame_high):
                    blended_val = (1 - weight) * f_low + weight * f_high
                    blended_features.append(blended_val)

                final_sequence.append({"features": blended_features})

        elif num_extracted_frames == 1:
            final_sequence = [video_frames_features[0]] * SEQUENCE_LENGTH
        else:
            st.error("No hands detected anywhere in this video file. Sending empty vector baseline.")
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
