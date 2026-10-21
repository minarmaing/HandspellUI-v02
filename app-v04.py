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

    if st.button("Predict"):
        letters = []
        columns = st.columns(len(ordered_files))

        for column, file in zip(columns, ordered_files):
            with column:
                try:
                    image = Image.open(file)
                    image.load()
                    st.image(image, caption=file.name, width=120)

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
                    response.raise_for_status()
                    result = response.json()

                    letter = result["prediction"]
                    confidence = result.get("confidence")

                    if letter in ("NO HAND", "NO HAND/SIGN"):
                        letter = "?"
                        st.warning("No hand detected")
                    else:
                        st.subheader(letter)

                    letters.append(letter)

                    if confidence is not None:
                        st.caption(f"Score: {confidence:.1%}")

                except (
                    requests.RequestException,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as e:
                    letters.append("?")
                    st.error(str(e))

        st.subheader("Predicted word: " + "".join(letters))
