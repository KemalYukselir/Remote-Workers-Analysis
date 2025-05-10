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

  st.markdown(
      """
      # 🌍 **Remote Worker Analysis** 🌍
      ## By **Kemal Yukselir**

      ### **Description:**
      This project explores workplace survey data to understand what influences whether individuals seek support for burnout. 
      It uses data analysis, decision tree modelling, and comment analysis to highlight how personal and workplace conditions relate to burnout support.

      ### **References:**
      - [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

      ### **Modules Used:**
      - **Pandas**  
      - **NumPy**  
      - **Scikit-learn**  
      - **Xgboost**  
      - **GridSearchCV**  
      - **Streamlit**  
      - **Matplotlib**  
      - **Seaborn**  
      - **Nltk**

      ### **Project Highlights:**
      With all ethical practices considered, this is the best model achieved after multiple iterations:

      - 🧪 **Training Accuracy:** 0.8514
      - 🧾 **Testing Accuracy:**  0.8649

      #### **📊 Classification Report:**
      | Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
      |----------------|---------|---------|----------|-----------|--------------|
      | **Precision**  | 0.79    | 0.97    |          | 0.88      | 0.88         |
      | **Recall**     | 0.97    | 0.77    |          | 0.87      | 0.86         |
      | **F1-Score**   | 0.87    | 0.86    | 0.86     | 0.86      | 0.86         |
      | **Support**    | 35      | 39      | 74       |           |              |
      """
  )

if page == "Project Overview":
    project_overview_page()