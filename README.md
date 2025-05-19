# 🧑‍💻 Remote Worker Burnout Analysis & Predictor

## Live Demo
- [Streamlit App](https://remote-workers-analysis.streamlit.app/) 

### ✍️ By Kemal Yukselir

## How to Run
- Install requirements: `pip install -r requirements.txt`
- Run the app: `streamlit run main.py`

# Update Venve modules
- Pip freeze > requirements.txt
- pigar generate 

## Project Description
This project analyses and predicts burnout risk among remote workers using survey data. It combines data cleaning, data analysis, feature engineering, XGBoost modeling, and NLP comment analysis to:
- Identify key factors influencing burnout
- Visualize insights with Tableau and Streamlit
- Provide a live predictor for burnout risk based on user input

## Dataset Reference
- [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- [About OSMI](https://osmihelp.org/)

## Key Features
- Data cleaning and feature engineering (see `EDARemoteWorkers.py`)
- Burnout prediction using XGBoost (see `ModelRemoteWorkerAnalysis.py`)
- NLP analysis of worker comments (see `NLPMentalHealth.py`)
- Streamlit dashboard with:
  - Project overview
  - Ethical standards
  - Interactive Tableau dashboard
  - Live burnout predictor

## Modules Used
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- nltk
- wordcloud
- streamlit
- joblib

## Project Highlights
- **Training Accuracy:** ~0.76
- **Testing Accuracy:** ~0.70
- **Balanced dataset:** ~50% burnout rate among remote workers
- **Key insights:**
  - Burnout is not limited to company type (tech/non-tech)
  - Smaller companies show higher burnout risk
  - Lack of wellness programs and support increases burnout
  - NLP shows 'company' is the most common word in burnout comments
- **Ethical standards:**
  - No personal identifiers used
  - Data is anonymized and public
  - Model and features are transparent and open source

## How it Works
- Data is cleaned and preprocessed (see `EDARemoteWorkers.py`)
- Model is trained and evaluated (see `ModelRemoteWorkerAnalysis.py`)
- NLP extracts common burnout themes from comments (see `NLPMentalHealth.py`)
- Streamlit app provides:
  - Project context and ethical considerations
  - Interactive Tableau dashboard
  - Burnout risk predictor based on user input

## Visualizations
- Tableau dashboard embedded in the app (`tableau_embed.html`)
- Matplotlib/Seaborn plots for EDA and NLP

## Contact
- [LinkedIn](https://www.linkedin.com/in/kemal-yukselir/)
- [Email](mailto:K.Yukselir123@gmail.com)
- [GitHub](https://github.com/KemalYukselir/Remote-Workers-Analysis)


