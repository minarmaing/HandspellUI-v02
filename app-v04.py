from PIL import Image, UnidentifiedImageError

import streamlit as st
import requests

API_BASE_URL = st.secrets["API_URL"]

st.set_page_config(page_title="HandSpell Reader", layout="centered")
st.title("✋ HandSpell Reader")
st.write("Upload a hand sign image to predict the letter.")

uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True,)

if uploaded_files:
    ordered_files = sorted(uploaded_files, key=lambda f: f.name)
    st.write("Image order:", " → ".join(f.name for f in ordered_files))

    if st.button("Predict"):
        letters = []
        for file in ordered_files:
            st.image(
                file.getvalue(),
                caption=file.name,
                width=150,
            )

            try:
                image = Image.open(file)
                image.load()
                st.image(image, width=300)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("Cannot read this image. Please upload another image.")
                st.stop()

    try:
        with st.spinner("Classifying..."):
            for file in ordered_files:
                response = requests.post(
                    API_BASE_URL,
                    files={
                        "file": (
                            file.name,
                            file.getvalue(),
                            file.type,
                        )
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    result = response.json()

                    letter = result["prediction"]
                    confidence = result.get("confidence")

                    if letter in ("NO HAND", "NO HAND/SIGN"):
                        letters.append("?")
                        st.warning("No hand/sign detected.")
                    else:
                        letters.append(letter)
                        st.write(f"Letter: {letter}")

                    if confidence is not None:
                        st.write(f"Model score: {confidence:.1%}")

                    # letters.append(letter)

                    st.success(f"Predicted letter: {letter}")
                    if confidence is not None:
                        st.write(f"Model score: {confidence:.1%}")
                    st.json(result)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        letters.append("?")
        st.error(f"Could not predict {file.name}: {e}")

    st.subheader("Result: " + "".join(letters))
