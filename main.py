import streamlit as st
# Set page to perma wide
st.set_page_config(layout="wide")
import numpy as np
from ModelRemoteWorkerAnalysis import ModelRemoteWorkerAnalysis

@st.cache_resource
def load_model():
    return ModelRemoteWorkerAnalysis()

# Preload cached resources
model = load_model()