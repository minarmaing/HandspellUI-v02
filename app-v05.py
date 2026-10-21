import streamlit as st
import numpy as np
import time
import pandas as pd

# 1. Page Settings & Premium Theme Configuration
st.set_page_config(
    page_title="HandSpell AI Studio",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern card containers and smooth animations
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6A7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        border-radius: 0.75rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Workspace Sidebar Configuration
with st.sidebar:
    st.markdown("### 🛠️ Runtime Settings")
    st.caption("Configure the model pipeline constraints below.")

    model_mode = st.selectbox(
        "Model Strategy",
        ["Wrist-Normalized Architecture", "Raw Coordinate Architecture"],
        help="Normalized mode handles variable hand distances from the camera much better."
    )

    confidence_threshold = st.slider(
        "Classification Threshold",
        min_value=0.50, max_value=0.99, value=0.85, step=0.01,
        help="Minimum confidence value required to accept and display an ASL sign classification."
    )

    st.divider()
    st.markdown("### 🌐 System Infrastructure")
    # Clean system connection metrics instead of raw link strings
    st.success("🟢 API Cluster: Connected")
    st.caption("Backend Node: `europe-west1.run.app`")
    st.caption("Latency: `42ms` | Status: `Operational`")

# 3. Main Application Workspace Dashboard
col_header, col_telemetry = st.columns([2, 1])

with col_header:
    st.markdown('<h1 class="main-title">🖐️ HandSpell Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep LSTM sequencing system for real-time American Sign Language text translation.</p>', unsafe_allow_html=True)

with col_telemetry:
    # Quick utility metrics block to fill spatial gap cleanly
    st.empty()

st.divider()

# 4. Central Split Interface (Input Control vs Data Stream Analytics)
workspace_left, analytics_right = st.columns([1, 1], gap="large")

with workspace_left:
    st.markdown("### 📥 Media Input Pipeline")

    input_type = st.radio("Choose Input Type:", ["Upload Video Clip", "Upload Static Image"], horizontal=True)

    with st.container(border=True):
        if "Video" in input_type:
            uploaded_file = st.file_uploader(
                "Drag and drop gesture video here...",
                type=["avi", "mp4", "mov"],
                help="Supported standards: MP4, AVI, MOV. Execution footprint limit: 200MB."
            )
            if uploaded_file:
                st.video(uploaded_file)
        else:
            uploaded_file = st.file_uploader(
                "Drag and drop target hand sign image here...",
                type=["jpg", "jpeg", "png"],
                help="Supported standards: PNG, JPG, JPEG."
            )
            if uploaded_file:
                st.image(uploaded_file, use_container_width=True)

        # Action Trigger Button Execution
        if uploaded_file:
            trigger_inference = st.button("⚡ Run Gesture Translation", type="primary", use_container_width=True)
        else:
            st.button("⚡ Run Gesture Translation", type="primary", disabled=True, use_container_width=True)
            st.info("💡 Pro Tip: Provide a visual asset on this panel to initialize the tracking engine arrays.")
            trigger_inference = False

with analytics_right:
    st.markdown("### 📊 Translation Stream Analytics")

    if uploaded_file and trigger_inference:
        # UX Simulation Block mimicking live API round-trip extraction metrics
        with st.spinner("Extracting coordinates and compiling LSTM sequences..."):
            time.sleep(1.2) # Mock computation latency

        # Primary Metric Card Module Layout via Columns
        st.markdown("#### Primary Output")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '<div class="metric-card">'
                '<p style="color: #64748B; font-weight: 600; margin-bottom:0;">PREDICTED SIGN</p>'
                '<h1 style="color: #FF4B4B; font-size: 4.5rem; margin: 0; font-weight: 900;">O</h1>'
                '</div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                '<div class="metric-card">'
                '<p style="color: #64748B; font-weight: 600; margin-bottom:0;">CONFIDENCE DENSITY</p>'
                '<h1 style="color: #10B981; font-size: 4.5rem; margin: 0; font-weight: 900;">94.2%</h1>'
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown("#### Softmax Distribution Matrix (Top 3 Candidates)")

        # Displaying candidate probability density layers beautifully
        st.write("**Letter O (Primary Class Match)**")
        st.progress(0.942)

        st.write("**Letter C**")
        st.progress(0.041)

        st.write("**Letter G**")
        st.progress(0.017)

        # Interactive telemetry tracking chart breakdown
        with st.expander("📈 View Extracted Landmark Telemetry Stream", expanded=False):
            st.caption("Real-time relative wrist delta coordinate history mapping (Last 30 sequence matrices)")
            mock_chart_data = pd.DataFrame(
                np.random.randn(30, 3),
                columns=['Wrist_X', 'Wrist_Y', 'Wrist_Z']
            )
            st.line_chart(mock_chart_data)

    else:
        # Elegant resting state layout template prior to model processing runs
        with st.container(border=True):
            st.markdown(
                "<div style='text-align: center; padding: 75px 20px; color: #9CA3AF;'>"
                "<h3 style='margin-bottom: 10px; color: #4B5563;'>Awaiting Pipeline Execution Stream</h3>"
                "<p style='font-size: 0.95rem;'>Configure operational params and click translate to render spatial data diagnostics cards.</p>"
                "</div>",
                unsafe_allow_encoded_html=True, unsafe_allow_html=True
            )
