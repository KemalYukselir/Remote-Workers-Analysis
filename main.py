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

def project_overview_page():
  st.title("📘 Project Overview 📘")

  st.markdown("""
  # 🌍 Remote Worker Analysis 🌍
  ## By Kemal Yukselir

  ### Description:
  This project explores workplace survey data to understand what influences whether individuals seek support for burnout. 
  It uses data analysis, decision tree modelling, and comment analysis to highlight how personal and workplace conditions relate to burnout support.
              
  ### References
  - [OSMI](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

  ### Modules:
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - Statsmodels  
  - Category Encoders  
  - Streamlit  
  - Matplotlib  
  - Seaborn
  - itertools

  ### Project Highlights:
  With all ethical practise considered, this is the best model I can get with many reruns.

  - 🧪 Training Accuracy: 0.8514
  - 🧾 Testing Accuracy:  0.8649
              
  - 📊 Classification Report:
                precision    recall  f1-score   support

            0       0.79      0.97      0.87        35
            1       0.97      0.77      0.86        39

    accuracy                           0.86        74
    macro avg       0.88      0.87      0.86        74
    weighted avg    0.88      0.86      0.86        74
  """)

if page == "Project Overview":
    project_overview_page()