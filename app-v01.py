from PIL import Image, UnidentifiedImageError

import streamlit as st
import requests

API_BASE_URL = st.secrets["API_URL"]

st.set_page_config(page_title="HandSpell Reader", layout="centered")
st.title("✋ HandSpell Reader")
st.write("Upload a hand sign image to predict the letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
        st.image(image, width=300)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Cannot read this image. Please upload another image.")
        st.stop()

    try:
        with st.spinner("Classifying..."):
            response = requests.post(
                API_BASE_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
                timeout=120,
            )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted letter: {result['prediction']}")
            st.write(f"Model score: {result['confidence']:.1%}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except (ValueError, KeyError, TypeError):
        st.error("The prediction service returned an invalid response.")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.stop()
