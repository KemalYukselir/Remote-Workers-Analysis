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

def ethical_standards_page():
    st.title("📄 Project Ethical Standards 📄")

    st.markdown("""
    ## Responsible Use of Machine Learning in Burnout Support Analysis

    **Overview**  
    - This project uses self-reported survey data to explore which factors influence whether individuals seek support for burnout.  
    - While models like this can raise awareness and guide supportive policies, they must be used responsibly.

    ### ⚖️ Key Ethical Considerations

    - **Bias in Data**  
      - The dataset reflects personal and workplace experiences that may vary across regions and cultures.  
      - Categorical variables such as gender and age are handled carefully to avoid reinforcing stereotypes.  
      - The model is not intended for individual prediction or decision-making.

    - **Data Privacy**  
      - The data used is anonymised and publicly available. No personally identifiable information is included.  
      - The open-ended comment data is analysed for general themes only, not linked to individuals.

    - **Accountability**
      - If you have any concerns or suggestions regarding the ethical implications of this project:
        - [LinkedIn](https://www.linkedin.com/in/kemal-yukselir/)
        - [Email Me](https://mail.google.com/mail/u/0/?fs=1&to=K.Yukselir123@gmail.com&tf=cm)

    - **Transparency & Interpretability**  
      - The model used (Decision Tree) was chosen for its transparency and ease of interpretation.  
      - Head over to the Project Overview for details on how features were selected and evaluated.
      - This is an open source project available on [GitHub](https://github.com/KemalYukselir/Remote-Workers-Analysis)


    - **Intended Use**  
      - This project is for **educational and analytical purposes only**.  
      - It is meant to explore trends in burnout support-seeking behaviour — not to diagnose or classify individuals.  
      - It should not be used for employment decisions, profiling, or mental health evaluation.

    ### 📚 Further Reading

    - [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    - [Ethics in AI and Data Science (UK Government Guide)](https://www.gov.uk/government/publications/data-ethics-framework/data-ethics-framework-2020)
    """)


if page == "Project Overview":
    project_overview_page()
elif page == "Ethical Standards":
    ethical_standards_page()