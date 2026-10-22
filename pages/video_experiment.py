import streamlit as st
import cv2
import mediapipe as mp
import requests
import numpy as np
import tempfile
import os

st.warning("Experimental prototype — accuracy is being improved.")

if st.button("← Back to HandSpell Studio"):
    st.switch_page("app-v02.py")
