
import streamlit as st
import cv2
import numpy as np
import os
import json
import urllib.request
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="ASL Gesture Recognition UI", layout="wide")

st.title("🖐️ Real-time & Video ASL Gesture Recognition")
st.write("Upload a short video clip (AVI/MP4) of an ASL sign to classify it using your trained LSTM models.")

# Helper to download MediaPipe task if missing
def download_landmark_model():
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        with st.spinner("Downloading Hand Landmarker Task model..."):
            urllib.request.urlretrieve(model_url, model_path)
    return model_path

# Normalize sequence coordinates
def normalize_sequence_features(sequence_tensor):
    norm_seq = sequence_tensor.copy()
    frame = norm_seq[0]
    for frame_idx in range(frame.shape[0]):
        # Left Hand (landmarks 0-20, features 0-63)
        lh_wrist_x, lh_wrist_y, lh_wrist_z = frame[frame_idx, 0], frame[frame_idx, 1], frame[frame_idx, 2]
        if lh_wrist_x != 0.0 or lh_wrist_y != 0.0 or lh_wrist_z != 0.0:
            for i in range(21):
                frame[frame_idx, i*3] -= lh_wrist_x
                frame[frame_idx, i*3 + 1] -= lh_wrist_y
                frame[frame_idx, i*3 + 2] -= lh_wrist_z

        # Right Hand (landmarks 0-20, features 63-125)
        rh_wrist_x, rh_wrist_y, rh_wrist_z = frame[frame_idx, 63], frame[frame_idx, 64], frame[frame_idx, 65]
        if rh_wrist_x != 0.0 or rh_wrist_y != 0.0 or rh_wrist_z != 0.0:
            for i in range(21):
                base = 63 + i*3
                frame[frame_idx, base] -= rh_wrist_x
                frame[frame_idx, base + 1] -= rh_wrist_y
                frame[frame_idx, base + 2] -= rh_wrist_z
    return norm_seq

# Extract coordinates from video
def extract_landmarks(video_path, model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    standard_features = []
    for hand in ['left_hand', 'right_hand']:
        for i in range(21):
            for coord in ['x', 'y', 'z']:
                standard_features.append(f"{coord}_{hand}_{i}")
    feature_indices = {feat: idx for idx, feat in enumerate(standard_features)}
    sequence_frames = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)
        frame_features = np.zeros(126)

        if detection_result.hand_landmarks and detection_result.handedness:
            for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
                label = handedness[0].category_name.lower()
                hand_prefix = "left_hand" if "left" in label else "right_hand"
                for idx, landmark in enumerate(hand_landmarks):
                    x_name, y_name, z_name = f"x_{hand_prefix}_{idx}", f"y_{hand_prefix}_{idx}", f"z_{hand_prefix}_{idx}"
                    if x_name in feature_indices: frame_features[feature_indices[x_name]] = landmark.x
                    if y_name in feature_indices: frame_features[feature_indices[y_name]] = landmark.y
                    if z_name in feature_indices: frame_features[feature_indices[z_name]] = landmark.z
        sequence_frames.append(frame_features)

    cap.release()
    seq_matrix = np.array(sequence_frames)
    if len(seq_matrix) == 0:
        seq_matrix = np.zeros((30, 126))
    elif len(seq_matrix) >= 30:
        seq_matrix = seq_matrix[:30]
    else:
        seq_matrix = np.vstack((seq_matrix, np.zeros((30 - len(seq_matrix), 126))))

    return np.expand_dims(seq_matrix, axis=0)

# Sidebar options
st.sidebar.header("Configuration Panel")
model_type = st.sidebar.selectbox("Choose Trained Model:", ["Wrist-Normalized Model", "Raw Coordinates Model"])

# Define files based on selection
if model_type == "Wrist-Normalized Model":
    model_file = "models/asl_alphabet_normalized_lstm.keras"
    meta_file = "models/asl_alphabet_normalized_metadata.json"
    use_norm = True
else:
    model_file = "models/asl_alphabet_lstm.keras"
    meta_file = "models/asl_alphabet_normalized_metadata.json"
    use_norm = False

if not os.path.exists(model_file):
    st.error(f"Model path `{model_file}` not found. Ensure the model has been trained and saved successfully.")
else:
    # Load labels
    classes = []
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
            classes = meta_data.get("classes", [])
    else:
        classes = [chr(i) for i in range(65, 91)] # fallback default A-Z

    # Upload Section
    uploaded_file = st.file_uploader("Choose a gesture video file...", type=["avi", "mp4", "mov"])

    if uploaded_file is not None:
        # Save file locally
        temp_video_path = "temp_uploaded_video.avi"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(uploaded_file)

        if st.button("⚡ Run Gesture Inference"):
            landmark_model_path = download_landmark_model()
            
            with st.spinner("Processing video frames and extracting coordinates..."):
                raw_sequence = extract_landmarks(temp_video_path, landmark_model_path)
                
            if use_norm:
                final_sequence = normalize_sequence_features(raw_sequence)
                st.info("Wrist-Normalization strategy applied successfully.")
            else:
                final_sequence = raw_sequence

            # Inference
            with st.spinner("Executing LSTM prediction..."):
                model = tf.keras.models.load_model(model_file)
                predictions = model.predict(final_sequence)[0]

            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            predicted_label = classes[predicted_idx]

            # Visual results display
            st.success(f"### Predicted Gesture: **{predicted_label}** (Confidence: {confidence:.2%})")
            
            # Probability distribution visualizers
            st.write("#### Top 3 Gesture Candidates:")
            top_3_indices = np.argsort(predictions)[::-1][:3]
            for idx in top_3_indices:
                label = classes[idx]
                prob = predictions[idx]
                st.write(f"**{label}** ({prob:.2%})")
                st.progress(float(prob))

            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
