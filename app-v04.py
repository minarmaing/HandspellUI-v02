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

        with st.spinner("Classifying..."):
            for file in ordered_files:
                try:
                    image = Image.open(file)
                    image.load()
                    st.image(image, caption=file.name, width=150)

                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        files={
                            "file": (
                                file.name,
                                file.getvalue(),
                                file.type,
                            )
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()

                    letter = result["prediction"]
                    confidence = result.get("confidence")

                    if letter in ("NO HAND", "NO HAND/SIGN"):
                        letters.append("?")
                        st.warning(f"{file.name}: No hand/sign detected.")
                    else:
                        letters.append(letter)
                        st.write(f"Letter: {letter}")

                        if confidence is not None:
                            st.write(f"Model score: {confidence:.1%}")

                except (
                    requests.RequestException,
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as e:
                    letters.append("?")
                    st.error(f"Could not predict {file.name}: {e}")

        st.subheader("Result: " + "".join(letters))
