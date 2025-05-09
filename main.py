import streamlit as st
# Set page to perma wide
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
from ModelRemoteWorkerAnalysis import ModelRemoteWorkerAnalysis

@st.cache_resource
def load_model():
    return ModelRemoteWorkerAnalysis()

# Preload cached resources
model = load_model()

# Side bar title
st.sidebar.title("Remote Worker Analysis")

# Sidebar Navigation
page = st.sidebar.selectbox("📂 Select a Page", ["Project Overview", "Ethical Standards", "Insights", "Predictor"])

# Sidebar Image
st.sidebar.image("assets/RemoteWorkImg.jpeg", width=900)

# Added direction to other project
st.sidebar.markdown("**Checkout my other project:**")
st.sidebar.markdown("[Student Certify Rate](https://student-certify-rate.streamlit.app/)")

