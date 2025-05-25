# Regex
import re
import os
import praw
from dotenv import load_dotenv
import praw

# Conntect to api
import requests # Connect to external websites
from PIL import Image # display images
import urllib.request # Similar to requests
# from gensim.corpora import Dictionary # Mapping of all english words

# Big 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # Create word clouds

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

# Simple test: fetch 5 hot posts from r/Python
for post in reddit.subreddit("Python").hot(limit=5):
    print(f"Title: {post.title}")
    print(f"Upvotes: {post.score}")
    print("---")
