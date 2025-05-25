# Regex
import re
import os
import praw
from dotenv import load_dotenv

# Conntect to api
import requests # Connect to external websites
from PIL import Image # display images

# Big 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Natural language Tokeniser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download text files to reuse
# nltk.download('stopwords') # Download list of stop words
# nltk.download('punkt_tab') # Download a list of punctuation
# nltk.download('punkt') # Download a list of punctuation

# Load env variables
load_dotenv()

# Replace these with your actual credentials
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),  
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT'))

# Define the subreddit and search query
subreddit_name = "remotework"
query = "work remote"

# Search the subreddit
results = reddit.subreddit(subreddit_name).search(query, sort="new", limit=20)

# Collect results as a list of dicts for proper DataFrame structure
clean_results = []

for post in results:
    clean_results.append({
        'texts': post.title + " " + post.selftext[:400]
    })

# Transform into dataframe with columns 'title' and 'selftext'
df = pd.DataFrame(clean_results)