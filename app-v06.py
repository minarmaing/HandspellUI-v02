"""HandSpell demo. Start with: python -m streamlit run app.py"""

import hashlib
import html
import io
import math
import os
import time
from urllib.parse import urlsplit, urlunsplit

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

NO_HAND = {"NO HAND", "NO HAND/SIGN", "NO_HAND"}
ALPHABET = set("ABCDEFGHIKLMNOPQRSTUVWXY")  # Static letters: excludes J and Z.
MAX_BYTES = 10 * 1024 * 1024


def prediction_url(value):
    """Accept either the API base URL or the complete /predict URL."""
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("API_URL must be an http:// or https:// URL.")
    path = parts.path.rstrip("/")
    if not path.endswith("/predict"):
        path += "/predict"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def normalize_result(payload):
    """Support the two response keys already used by the team's APIs."""
    if not isinstance(payload, dict):
        raise ValueError("API response must be a JSON object.")
    letter = payload.get("prediction") or payload.get("letter")
    if not isinstance(letter, str) or not letter.strip():
        raise ValueError("API response needs a 'prediction' or 'letter' string.")
    letter = letter.strip().upper()
    no_hand = letter in NO_HAND
    if not no_hand and letter not in ALPHABET:
        raise ValueError(f"Unsupported prediction: {letter}")
    confidence = payload.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Confidence must be a number between 0 and 1, or null.")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
    return {"letter": letter, "confidence": confidence, "no_hand": no_hand}


def read_image(upload):
    raw = upload.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError("Please use an image smaller than 10 MB.")
    with Image.open(io.BytesIO(raw)) as original:
        original.load()
        image = ImageOps.exif_transpose(original).convert("RGB")
    # Preview and inference use the same orientation and RGB pixels.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


def request_prediction(url, image_bytes):
    start = time.perf_counter()
    response = requests.post(
        url,
        files={"file": ("handspell.png", image_bytes, "image/png")},
        timeout=(10, 120),
    )
    response.raise_for_status()
    payload = response.json()
    return {
        **normalize_result(payload),
        "top_candidates": payload.get("top_candidates", []),
        "raw": payload,
        "elapsed": time.perf_counter() - start,
        "url": response.url,
    }


def accepted_letter(result, threshold):
    if result.get("error") or result["no_hand"]:
        return "?"
    score = result["confidence"]
    return "?" if score is not None and score < threshold else result["letter"]


def render_result(result, threshold):
    if result.get("error"):
        st.error(result["error"])
        return
    letter = accepted_letter(result, threshold)
    if result["no_hand"]:
        st.warning("No hand/sign detected. Try a clear photo of the whole hand.")
    elif letter == "?":
        st.warning("Below your display threshold. Try another photo.")
    a, b = st.columns(2)
    with a:
        st.metric("Predicted sign", result["letter"] if not result["no_hand"] else "—")
    with b:
        score = result["confidence"]
        st.metric("Model score", "—" if score is None else f"{score:.1%}")
    if result["confidence"] is not None:
        st.progress(float(result["confidence"]))
    st.caption(f"Request completed in {result['elapsed']:.2f} s")
    with st.expander("API response"):
        st.caption(result["url"])
        st.json(result["raw"])


def main():
    st.set_page_config(page_title="HandSpell Studio", page_icon="✋", layout="wide")
    st.markdown(
        """<style>
      .stApp {background:#ffffff; color:#283142;}
      [data-testid="stSidebar"] {background:#f0f3f8;}
      h1 {color:#ff535b !important;}
      div[data-testid="stMetric"] {background:#f7f9fc; padding:22px;
          border:1px solid #e5eaf2; border-radius:14px; min-height:150px;}
      div[data-testid="stMetricValue"] {color:#ff535b; font-size:2.8rem;}
      .word {padding:24px; background:#f7f9fc; border:1px solid #e5eaf2;
          border-radius:14px; color:#ff535b; font-size:2.5rem;
          font-weight:700; letter-spacing:.12em; overflow-wrap:anywhere;}
    </style>""",
        unsafe_allow_html=True,
    )

    configured_url = os.getenv("API_URL", "")
    if not configured_url:
        try:
            configured_url = st.secrets["API_URL"]
        except (FileNotFoundError, KeyError):
            configured_url = "http://localhost:8000"

    with st.sidebar:
        st.header("⚙️ Demo settings")
        mode = st.radio("Translation mode", ["Single letter", "Spell a word"])
        threshold = st.slider(
            "Minimum displayed score",
            0.0,
            1.0,
            0.0,
            0.05,
            help="Below this score, the word uses ?. This does not change the model.",
        )
        st.caption("Model scores are not a guarantee of correctness.")
        st.divider()
        st.subheader("Backend")
        st.caption(configured_url)
        st.caption("Connection is verified when a prediction request succeeds.")
        st.divider()
        st.caption("24 static ASL letters · J and Z require movement.")

    try:
        url = prediction_url(configured_url)
    except (ValueError, AttributeError) as exc:
        st.error(f"Invalid API_URL configuration: {exc}")
        st.stop()

    st.title("✋ HandSpell Studio")
    st.write("Read a hand sign, or assemble a word from a sequence of hand images.")
    st.divider()
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.header("📥 Image input")
        st.caption("Use good lighting and keep the whole hand and wrist visible.")
        if mode == "Single letter":
            source = st.radio(
                "Input", ["Upload image", "Take a photo"], horizontal=True
            )
            upload = (
                st.file_uploader(
                    "Choose an image", type=["jpg", "jpeg", "png"], key="single_upload"
                )
                if source == "Upload image"
                else st.camera_input("Take a hand sign photo")
            )
            files = [upload] if upload is not None else []
        else:
            source = "Upload images"
            st.info(
                "Number files in word order: 01_T.jpg, 02_I.jpg, 03_G.jpg, 04_E.jpg, 05_R.jpg."
            )
            files = st.file_uploader(
                "Choose images (maximum 12)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="word_upload",
            )
            files = sorted(files, key=lambda f: f.name.lower())
            if len(files) > 12:
                st.warning("Please select at most 12 images for this demo.")
                st.stop()

        signature = (
            url,
            mode,
            source,
            tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files),
        )
        if st.session_state.get("input_signature") != signature:
            st.session_state["input_signature"] = signature
            st.session_state["results"] = []

        prepared = []
        for file in files:
            try:
                image, encoded = read_image(file)
                prepared.append({"name": file.name, "image": image, "bytes": encoded})
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as exc:
                st.error(f"Cannot read {file.name}: {exc}")

        if mode == "Single letter" and prepared:
            st.image(prepared[0]["image"], caption=prepared[0]["name"], width=420)
        elif prepared:
            st.write("Order:", " → ".join(item["name"] for item in prepared))

        run = st.button(
            "⚡ Translate" if mode == "Single letter" else "⚡ Translate word",
            type="primary",
            disabled=not prepared or len(prepared) != len(files),
        )
        if run:
            results = []
            with st.spinner("Reading hand signs…"):
                for item in prepared:
                    try:
                        result = request_prediction(url, item["bytes"])
                    except (requests.RequestException, ValueError) as exc:
                        result = {"error": f"Prediction failed: {exc}"}
                    results.append({"name": item["name"], **result})
            st.session_state["results"] = results

    results = st.session_state.get("results", [])
    with right:
        st.header("📊 Translation results")
        if not results:
            st.info("Choose your images and select Translate.")
        elif mode == "Single letter":
            render_result(results[0], threshold)

            candidates = results[0].get("top_candidates", [])[:3]

            if candidates:
                st.markdown("#### Top 3 candidates")

                cards = []

                for rank, candidate in enumerate(candidates, start=1):
                    letter = html.escape(str(candidate["letter"]))
                    score = float(candidate["confidence"])
                    percentage = max(0.0, min(score, 1.0)) * 100

                    winner = rank == 1
                    background = "#FFF1ED" if winner else "#F8FAFC"
                    border = "#E86751" if winner else "#E2E8F0"
                    color = "#C94F3D" if winner else "#475569"
                    label = "BEST MATCH" if winner else f"#{rank} CANDIDATE"

                    cards.append(f"""
                        <div style="
                        flex:1; min-width:100px;
                        padding:18px; border-radius:14px;
                        background:{background};
                        border:1px solid {border};">
                        <div style="
                            font-size:11px; font-weight:600;
                            letter-spacing:1px; color:{color};">
                            {label}
                        </div>
                        <div style="
                            font-size:42px; font-weight:700;
                            line-height:1.3; margin-top:10px;
                            color:{color};">
                            {letter}
                        </div>
                        <div style="
                            font-size:20px; font-weight:600;
                            margin:4px 0 14px; color:{color};">
                            {percentage:.1f}%
                        </div>
                        <div style="
                            height:5px; background:#E2E8F0;
                            border-radius:5px; overflow:hidden;">
                            <div style="
                                width:{percentage}%; height:100%;
                                background:{border};">
                            </div>
                        </div>
                    </div>
                    """)

                    st.markdown(
                        '<div style="display:flex; flex-wrap:wrap; gap:12px;">'
                        + "".join(cards)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
        else:
            word = "".join(accepted_letter(r, threshold) for r in results)
            st.subheader("Predicted word")
            st.markdown(
                f'<div class="word">{html.escape(word)}</div>', unsafe_allow_html=True
            )
            st.caption(
                "? means no detection, a failed request, or a score below your threshold."
            )
            st.caption(
                "Letters are joined in image order; no spelling correction is applied."
            )

    if mode == "Spell a word" and prepared:
        st.divider()
        st.subheader("Your sign sequence")
        # At most five cards per row keeps longer words readable.
        for start in range(0, len(prepared), 5):
            columns = st.columns(min(5, len(prepared) - start))
            for index, column in enumerate(columns, start=start):
                item = prepared[index]
                with column:
                    st.image(item["image"], caption=item["name"], width=140)
                    if index < len(results):
                        r = results[index]
                        if r.get("error"):
                            st.error(r["error"])
                        else:
                            st.subheader(accepted_letter(r, threshold))
                            if not r["no_hand"]:
                                st.caption(f"Model prediction: {r['letter']}")
                            else:
                                st.caption("No hand/sign detected")
                            if r["confidence"] is not None:
                                st.caption(f"Score: {r['confidence']:.1%}")
        if results:
            with st.expander("All API responses"):
                st.json(results)

    st.divider()
    st.subheader("🎬 Coming next: Recognition from video")

    st.write(
        "Moving beyond individual images to recognise "
        "hand movements across a short video."
    )

    st.info(
        "Experimental prototype built with a separate model and API. "
        "Prediction accuracy is still being evaluated and improved."
    )

    with st.expander("Explore the next step"):
        st.markdown("""
            **Planned experience**

            1. Upload a short video of a hand sign.
            2. Track hand landmarks across video frames.
            3. Analyse the movement sequence.
            4. Predict the sign.

            **Potential extension:** recognise movement-based letters
            such as **J** and **Z**.
            """)

        if st.button("🎬 Open experimental video prototype"):
            st.switch_page("pages/app-v02.py")

        st.caption(
            "Current demo: static images and fingerspelled words. "
            "Next milestone: reliable recognition of moving signs."
        )


if __name__ == "__main__":
    main()
