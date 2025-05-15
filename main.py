import streamlit as st
# Set page to perma wide
st.set_page_config(layout="wide")
import pandas as pd
import pickle
import streamlit.components.v1 as components

@st.cache_resource
def load_model():
    # Load model
    with open("data/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
        return model

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
    # Set the title and image
    st.title("🌍 **Remote Worker Burnout Analysis** 🌍")

    st.image("https://media.licdn.com/dms/image/v2/C4E12AQFULp9_xFcyLg/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1621418267262?e=2147483647&v=beta&t=OAsQn-62BIbtXd04IhvIjhDj3Z2Bm_D_rvZl99NwFKs", width=250)
    
    st.markdown(
      
      """
        ## By **Kemal Yukselir**

        ## **Context:**
        <img src="https://st.depositphotos.com/37378518/60435/v/450/depositphotos_604351950-stock-illustration-light-bulb-icon-isolated-white.jpg" width="100"/>
        
        > _"Going through Digital Futures training for 3 months has been a great learning experience.  
        However, based on both my own experience and the experience of my peers, remote work has also been a significant challenge.  
        This inspired me to explore the topic of burnout among remote workers."_

        ### **Description:**
        This project explores workplace survey data to understand what influences whether remote workers experience burnout.  
        It uses data analysis, XGBoost modelling, and comment analysis to highlight how personal and workplace conditions relate to burnout risk and support.


        ### **References:**        
        <img src="https://osmihelp.org/assets/img/osmi-logo-big.png" width="150"/>

        - [About OSMI](https://osmihelp.org/)
        - [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

        ### **About The Dataset:**    
        - The survey was conducted by OSMI (Open Sourcing Mental Illness).
        - A record in this survey represents a worker in tech and non tech companies.
        - The dataset contains generic questions about themself and their workplace.
        - I have narrowed down the dataset to only inlcude remote workers for this project scope.

        ### **Modules Used:**
        - **Pandas**  
        - **NumPy**  
        - **Scikit-learn**  
        - **Xgboost**  
        - **RandomizedSearchCV**  
        - **Streamlit**  
        - **Matplotlib**  
        - **Seaborn**  
        - **Nltk**

        ### **Project Highlights:**
        With all ethical practices considered, this is the best model achieved after multiple iterations:

        - 🧪 **Training Accuracy:** 0.8151 🧪
        - 🧾 **Testing Accuracy:**  0.7973 🧾

        #### **📊 Classification Report:**
        | Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
        |----------------|---------|---------|----------|-----------|--------------|
        | **Precision**  | 0.76    | 0.83    |          | 0.80      | 0.80         |
        | **Recall**     | 0.83    | 0.77    |          | 0.80      | 0.80         |
        | **F1-Score**   | 0.79    | 0.80    | 0.80     | 0.80      | 0.80         |
        | **Support**    | 35      | 39      | 74       |           |              |
      """,
        unsafe_allow_html=True
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
    st.title("📊 Remote Workers Burnout Dashboard 📊")

    # Display the HTML in Streamlit
    components.html(load_html(), height=750, scrolling=False)

    st.markdown("""
    ### **Key Insights:**
    **Figure 1:**
    - First I wanted to check how big the burnout problem is.
    - An alarming 52.703% of Remote workers reported feeling burnout.
                
    **Figure 2:**
    - Before we dive in further, I want to highlight to employers that this problem affects them and the company as a whole. Not just the employee.
    - We can see out of those who burnout, most of them feel that it "often" and "sometimes" affects the work they do.
    - Employers addressing this issue will not only help the employee but also the company as a whole.
                    
    **Figure 3:**
    - Then I wanted to check if the type of company was the main reason for the burnout.
    - From the graphs, we can see that tech and non tech companies have nearly equal distribution of burnout. So we can rule out that the type of company is not the main reason for the burnout.
    - So it must be a company structural issue in the workplace.
                
    **Figure 4:**
    - Figure 4 shows the 10 most common words of burntout remote workers used in the comments section.
    - We can see that the word company is the most common word used.
    - This further reinforces the idea that burntout must be a more of a company structural issue
                
    **Figure 5:**
    - Figure 5 shows that on average, the less employees a comapny has, the more liekly they are to burnout.
    - Could this indicate that smaller companies have more pressure on their remote employees?
    - Or could it be that larger companies have more resources to help their remote employees?

    **Figure 6:**
    - Figure 6 shows that most remote employees who are burnt out, their company does not offer mental health as part of their employee wellness program.
    - This shows that companies should invest more in their employee wellness programs to avoid burnout.
    - This could be weekly gym membership incentives, or even just a simple coffee break with the team.
    
    **Figure 7:**
    - Figure 7 shows that most remote employees who are burnt out, their company does not provuide resources to seek and get help when avoiding burnout.
    - Just a simple check in with the emplyees can help guide them to the right resources.
                   
    ## **Conclusion:**
    There is significant burnout when it comes to remote working. Both employees, employers and the company as a whole need to be aware of this since it affects all parties.
    It has been concluded that the type of company is not the main reason for the burnout. It must be a structural issue in the workplace.
    Some key insights show that companies need to invest more in their complyee wellness programs and provide resources to seek help for remote workers.
    This could be weekly check ins, gym membership incentives, or even just a simple coffee break with the team.
                
    While remote working could benefit the employee and reduce the costs of the company, if it's not approached right, 
    it can go wrong for the employee and the company as a whole.
    """)


def further_insights_page():
    pass

def model_page():
    st.title("🧠 Burnout Support Predictor")

    st.subheader("Enter Your Information")

    work_interfere = st.selectbox("If you have a mental health condition, could it potentially interfere with your work?", [
        "Never", "Rarely", "Sometimes", "Often", "X"
    ])

    family_history = st.selectbox("Do you have a family history of mental illness?", [
        "Yes", "No"
    ])

    obs_consequence = st.selectbox("Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?", [
        "Yes", "No"
    ])

    # Note: gender is not part of your described dataset, but keeping it in case it's still relevant
    gender = st.selectbox("Gender", ["Male", "Female"])

    benefits = st.selectbox("Does your employer provide mental health benefits?", [
        "Yes", "No", "Don't know"
    ])

    care_options = st.selectbox("Do you know the options for mental health care your employer provides?", [
        "Yes", "No", "Don't know"
    ])

    coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", [
        "Yes", "No", "Some of them"
    ])

    wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", [
        "Yes", "No", "Don't know"
    ])

    seek_help = st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", [
        "Yes", "No", "Maybe"
    ])

    no_employees = st.selectbox("How many employees does your company or organization have?", [
        "1-5", "6-25", "26-100", "101-500", "More than 500"
    ])

    mental_health_consequence = st.selectbox("Do you think that discussing a mental health issue with your employer would have negative consequences?", [
        "Yes", "No", "Maybe"
    ])

    mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", [
        "Yes", "No", "Don't know"
    ])

    supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", [
        "Yes", "No", "Some of them"
    ])

    anonymity = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", [
        "Yes", "No", "Don't know"
    ])
    
    leave = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", [
        "Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"
        ])


    # Construct input DataFrame
    input_dict = {
        'work_interfere': [work_interfere],
        'family_history': [family_history],
        'obs_consequence': [obs_consequence],
        "Gender": [gender],
        'benefits': [benefits],
        'care_options': [care_options],
        'coworkers': [coworkers],
        'wellness_program': [wellness_program],
        'seek_help': [seek_help],
        'no_employees': [no_employees],
        'mental_health_consequence': [mental_health_consequence],
        'mental_vs_physical': [mental_vs_physical],
        'supervisor': [supervisor],
        'anonymity': [anonymity],
        'leave': [leave]
    }

    input_df = pd.DataFrame(input_dict)

    if st.button("Predict Burnout Support Seeking"):
        prediction = model.predict_from_model(input_df)
        print(prediction)
        st.success(f"Predicted class: {'This person will likely feel burnout' if prediction == 1 else 'This person will not feel burnout'}")


if page == "Project Overview":
    project_overview_page()
elif page == "Ethical Standards":
    ethical_standards_page()
elif page == "Insights":
    insights_page()
# elif page == "Further Insights":
#     further_insights_page()
elif page == "Predictor":
    model_page()