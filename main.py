import streamlit as st
# Set page to perma wide
st.set_page_config(layout="wide")
import pandas as pd
import streamlit.components.v1 as components
import joblib

@st.cache_resource
def load_model():
    return joblib.load("data/xgb_model.pkl")

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
page = st.sidebar.selectbox("📂 Select a Page", ["Project Overview", "Ethical Standards", "Insights", 'Further Insights', "Predictor"])

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
        - I have narrowed down the dataset to only include remote workers for this project scope.

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

        - 🧪 **Training Accuracy:** 0.7568 🧪
        - 🧾 **Testing Accuracy:**  0.7027 🧾

        #### **📊 Classification Report:**
        | Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
        |----------------|---------|---------|----------|-----------|--------------|
        | **Precision**  | 0.69    | 0.72    |          | 0.70      | 0.70         |
        | **Recall**     | 0.69    | 0.72    |          | 0.70      | 0.70         |
        | **F1-Score**   | 0.69    | 0.72    | 0.70     | 0.70      | 0.70         |
      """,
        unsafe_allow_html=True
    )


def ethical_standards_page():
    st.title("📄 Project Ethical Standards 📄")

    st.markdown("""
    ## Responsible Use of Data Analysis and Machine Learning in Remote Workers Burnout Analysis

    ### ⚖️ Key Ethical Considerations

    - **Bias/Fairness in Data**  
      - The dataset reflects personal experiences from different backgrounds and environments.
      - Personal identifier fields like gender, age and country have been dropped to avoid bias to certain groups.

    - **Data Privacy**  
      - The dataset is fully anonymised and publicly available.
      - No personal identifiers are used at any stage.
      - Comments are also anonymised and are summarised for theme analysis only.

    - **Transparency**  
      - The model used (XGBoost) was selected for its balance of performance and feature insight.
      - Features has been reviewed to ensure the model remains understandable.
      - This is an open source project, available on [GitHub](https://github.com/KemalYukselir/Remote-Workers-Analysis).
      - Github also contains step by step commit and documentation of the project.
                
    - **Accountability**
      - If you have any concerns or feedback about the ethical use of this project:
        - [LinkedIn](https://www.linkedin.com/in/kemal-yukselir/)
        - [Email Me](mailto:K.Yukselir123@gmail.com)
      - Ethical guidlines and documentation follows the GOV guidlines [Click here](https://www.gov.uk/government/publications/data-ethics-framework/data-ethics-framework-2020)

    - **Intended Use**  
      - This project is for **educational and insight generation purposes only**.
      - It is designed to explore trends to avoid burnout and support in remote work — not to diagnose or assess individuals.
      - It should not be used for hiring, monitoring, or formal mental health evaluation.

    ### 📚 Further Reading
    - [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    - [Ethics in AI and Data Science – UK Government](https://www.gov.uk/government/publications/data-ethics-framework/data-ethics-framework-2020)
    """)


def insights_page():
    st.title("📊 Remote Workers Burnout Dashboard 📊")

    # Display the HTML in Streamlit
    components.html(
        load_html(),
        height=750,
        scrolling=True
    )

    st.markdown("""
    ### **Key Insights:**
    **Figure 1:**
    - First I wanted to check how big the burnout problem is.
    - An alarming 52.703% of remote workers reported feeling burnout.
                
    **Figure 2:**
    - We can see out of those who burnout, most of them feel that it "often" and "sometimes" affects the work they do.
    - This is interesting because it highlights to employers that this problem affects them and the company as a whole, not just the employee.
    - Employers addressing this issue will help all parties in the workplace.
                    
    **Figure 3:**
    - Then I wanted to check if the type of company was the main reason for the burnout.
    - From the graphs, we can see that tech and non-tech companies have nearly equal distribution of burnout. So we can rule out that the type of company is not the main reason for the burnout.
    - So it must be a company structural or personal issue in the workplace.
                
    **Figure 4:**
    - Figure 4 shows the 10 most common words burnt out remote workers used in the comments section.
    - We can see that the word company is the most common word used.
    - This further reinforces the idea that burnout must be more of a company structural issue
                
    **Figure 5:**
    - Figure 5 shows that on average, the fewer employees a company has, the more likely they are to burnout.
    - Could this indicate that smaller companies have more pressure on their remote employees?
    - Or could it be that larger companies have more resources to help their remote employees?

    **Figure 6:**
    - Figure 6 shows that most remote employees who are burnt out, their company does not offer mental health as part of their employee wellness program.
    - This shows that companies should invest more in their employee wellness programs to avoid burnout.
    - This could be weekly gym membership incentives, or even just a simple coffee break with the team.
    
    **Figure 7:**
    - Figure 7 is very interesting as it shows the opposite of what I expected.
    - Out of those who are burnt out, most said yes to "employer provide mental health benefits?" and "Do you know the options for mental health?"
    - This is very interesting as maybe the employees don't push themselves to use the benefits.
    - Companies can add in weekly check-ins to see if the employees need extra help or to even open them up to using resources more.
                   
    ## **Conclusion:**
    There is significant burnout when it comes to remote working. Both employees, employers, and the company as a whole need to be aware of this since it affects all parties.
    It has been concluded that the type of company is not the main reason for the burnout. It must be a structural or personal issue in the workplace.
    Some key insights show that companies need to invest more in their employee wellness programs and provide resources to seek help for remote workers.
    This could be weekly check-ins, gym membership incentives, or even just a simple coffee break with the team.
                
    While remote working could benefit the employee and reduce the costs of the company, if it's not approached right, 
    it can go wrong for the employee and the company as a whole.
    """)


def further_insights_page():
    st.title("🧾 Further Insights into Remote Working 🧾")

    st.subheader("💬 Most Frequent Words Used by Remote Workers on Reddit🧾")

    # Show the uploaded word cloud image
    st.image("assets/reddit-remote-work.png", caption="Word Cloud: Most Frequent Words in Remote Worker Reddit", use_container_width=True)

    st.markdown("""
    ### **Observations:**
    This word cloud shows the most commonly used words from remote worker comments.

    - **“Company”** appears prominently, reinforcing the earlier insight that **burnout is often tied to **organisational structure**, not the type of company.
    - The words **“need”**, **“support”**, and **“help”** also show up often, suggesting that many workers feel underserved or unsupported in their roles.
    - Interestingly, terms like **“everyone”** and **“know”** suggest that people either want to feel less alone or more informed about their workplace support.

    ### **Why This Matters:**
    - These word patterns highlight **communication gaps**, **emotional strain**, and the **need for stronger support structures** in remote environments.
    - Employers could take these insights to **improve check-ins**, **offer better resources**, or **enhance mental health awareness**.

    ### 📌 **Takeaway:**
    Language matters. The words people use can **reveal workplace culture challenges**.  
    By analysing how remote workers speak about their experiences, we can design better policies, communication strategies, and wellness programs to reduce burnout.
    """)


def model_page():
    st.title("🧠 Burnout Predictor")
    st.subheader("Enter Your Information")

    tech_company = st.selectbox(
        "Do you work at a tech company?", 
        ["Yes", "No"], 
        index=1  # Default to "No"
    )

    obs_consequence = st.selectbox(
        "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?", 
        ["Yes", "No"],
        index=1

    )

    benefits = st.selectbox(
        "Does your employer provide mental health benefits?", 
        ["Yes", "No", "Don't know"],
        index=1  # Default to "No"
    )

    care_options = st.selectbox(
        "Do you know the options for mental health care your employer provides?", 
        ['Not sure', 'No', 'Yes']
    )

    coworkers = st.selectbox(
        "Would you be willing to discuss a mental health issue with your coworkers?", 
        ["Yes", "No", "Some of them"]
    )

    wellness_program = st.selectbox(
        "Has your employer ever discussed mental health as part of an employee wellness program?", 
        ["Yes", "No", "Don't know"],
    )

    seek_help = st.selectbox(
        "Does your employer provide resources to learn more about mental health issues and how to seek help?", 
        ["Don't know", "No", "Yes"],
    )

    no_employees = st.selectbox(
        "How many employees does your company or organization have?", 
        ["1-5", "6-25", "26-100", "100-500", "More than 1000"]
    )

    mental_health_consequence = st.selectbox(
        "Do you think that discussing a mental health issue with your employer would have negative consequences?", 
        ["Yes", "No", "Maybe"],
        index=1  # Default to "No"
    )

    phys_health_consequence = st.selectbox(
        "Do you think that discussing a physical health issue with your employer would have negative consequences?", 
        ["Yes", "No", "Maybe"]
    )

    mental_vs_physical = st.selectbox(
        "Do you feel that your employer takes mental health as seriously as physical health?", 
        ["Yes", "No", "Don't know"]
    )

    supervisor = st.selectbox(
        "Would you be willing to discuss a mental health issue with your direct supervisor(s)?", 
        ["Yes", "No", "Some of them"]
    )

    anonymity = st.selectbox(
        "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", 
        ["Yes", "No", "Don't know"],
        index=1  # Default to "No"
    )

    leave = st.selectbox(
        "How easy is it for you to take medical leave for a mental health condition?", 
        ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"]
    )

    # Construct input DataFrame
    input_dict = {
        'tech_company': [tech_company],
        'obs_consequence': [obs_consequence],
        'benefits': [benefits],
        'care_options': [care_options],
        'coworkers': [coworkers],
        'wellness_program': [wellness_program],
        'seek_help': [seek_help],
        'no_employees': [no_employees],
        'mental_health_consequence': [mental_health_consequence],
        'phys_health_consequence': [phys_health_consequence],
        'mental_vs_physical': [mental_vs_physical],
        'supervisor': [supervisor],
        'anonymity': [anonymity],
        'leave': [leave]
    }

    input_df = pd.DataFrame(input_dict)

    if st.button("Predict Burnout Support Seeking"):
        prediction = model.predict_from_model(input_df)
        final_verdict = prediction[0]
        st.success(
            f"Predicted Burnout: {'This person will likely feel burnout' if final_verdict == 0 else 'This person will not feel burnout'}"
        )



if page == "Project Overview":
    project_overview_page()
elif page == "Ethical Standards":
    ethical_standards_page()
elif page == "Insights":
    insights_page()
elif page == "Further Insights":
    further_insights_page()
elif page == "Predictor":
    model_page()