import streamlit as st
# Set page to perma wide
st.set_page_config(layout="wide")
import numpy as np
import pandas as pd
import joblib 
from ModelRemoteWorkerAnalysis import ModelRemoteWorkerAnalysis
import streamlit.components.v1 as components

@st.cache_resource
def load_model():
    return ModelRemoteWorkerAnalysis()

@st.cache_resource
# Read the local HTML file
def load_html():
    with open("tableau_embed.html", 'r', encoding='utf-8') as f:
        html_data = f.read()
    return html_data

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


def insights_page():
    st.title("📊 Remote Workers Burnout Dashboard")

    # Display the HTML in Streamlit
    components.html(load_html(), height=800, scrolling=True)

    st.markdown("""
    ### **Key Insights:**
    - Firstly, an alarming 52.703% of Remote workers reported feeling burnout.
    - Before we dive in further, I want to highlight to employers this problem affects them and the company as a whole. Not just the employee.
    - We can see out of those who burnout, most of them feel that it "often" and "sometimes" affects the work they do.
    - Then I wanted to check if the type of company was the main reason for the burnout.
    - From the graphs, we can see that Tech and non tech comapnies have nearly equal distribution of burnout. So we can rule out that the type of company is not the main reason for the burnout.
    - So it must be a company structural issue in the workpalce.
    
    - The rest of the visualisations show some key factors affecting burnout.
    - First Graph shows comapnies don't include burnout solutions in their wellness programs.
    - Second Graph shows companies doesnt provide resources to seek help.
    """)


def model_page():
    st.title("🧠 Burnout Support Predictor")

    st.subheader("Enter Your Information")

    country = st.selectbox("Country", [
        'United States', 'France', 'United Kingdom', 'Canada', 'Poland', 'Australia',
        'Germany', 'Russia', 'Costa Rica', 'Austria', 'Mexico', 'South Africa',
        'Ireland', 'Romania', 'Brazil', 'Uruguay', 'New Zealand', 'Netherlands',
        'Finland', 'Bosnia and Herzegovina', 'Hungary', 'Singapore', 'Japan', 'India',
        'Bulgaria', 'Croatia', 'Bahamas, The', 'Greece', 'China'
    ])

    work_interfere = st.selectbox("How often does burnout interfere with your work?", [
        "Never", "Rarely", "Sometimes", "Often", "N/A"
    ])

    family_history = st.selectbox("Do you have a family history of mental health issues?", [
        "Yes", "No"
    ])

    care_options = st.selectbox("Do you know what mental health care options your employer provides?", [
        "Yes", "No", "Don't know"
    ])

    benefits = st.selectbox("Does your employer offer mental health benefits?", [
        "Yes", "No", "Don't know"
    ])

    anonymity = st.selectbox("Is your anonymity protected when using support services?", [
        "Yes", "No", "Don't know"
    ])

    coworkers = st.selectbox("Would you discuss burnout with coworkers?", [
        "Yes", "No", "Some of them"
    ])

    phys_health_interview = st.selectbox("Would you mention physical health issues in an interview?", [
        "Yes", "No", "Maybe"
    ])

    obs_consequence = st.selectbox("Have you seen negative consequences for coworkers with burnout?", [
        "Yes", "No"
    ])

    mental_health_consequence = st.selectbox("Do you think disclosing burnout could have negative consequences?", [
        "Yes", "No", "Maybe"
    ])

    # Construct input DataFrame
    input_dict = {
        "Country": [country],
        "work_interfere": [work_interfere],
        "family_history": [family_history],
        "care_options": [care_options],
        "benefits": [benefits],
        "anonymity": [anonymity],
        "coworkers": [coworkers],
        "phys_health_interview": [phys_health_interview],
        "obs_consequence": [obs_consequence],
        "mental_health_consequence": [mental_health_consequence]
    }

    input_df = pd.DataFrame(input_dict)

    # Load your pre-trained model and encoder (optional)
    # model = joblib.load("your_model.pkl")
    # encoder = joblib.load("your_encoder.pkl")
    # input_encoded = encoder.transform(input_df)

    if st.button("Predict Burnout Support Seeking"):
        prediction = model.predict_from_model(input_df)
        print(prediction)
        st.success(f"Predicted class: {'This person will likely feel burnout' if prediction[0] == 1 else 'This person will not feel burnout'}")


if page == "Project Overview":
    project_overview_page()
elif page == "Ethical Standards":
    ethical_standards_page()
elif page == "Insights":
    insights_page()
elif page == "Predictor":
    model_page()