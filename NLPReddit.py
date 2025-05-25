# Regex
import re
import os
import praw
from dotenv import load_dotenv

# Conntect to api
import requests # Connect to external websites
from PIL import Image # display images
from gensim.corpora import Dictionary # Mapping of all english words

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

def regex_clean(txt, regex):
    """Replace any text matching the regex

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove matches
    regex : string
        A text string of the regex pattern you want to match

    Returns
    -------
    The same txt string with the matches removes
    """

    return " ".join(re.sub(regex, "", txt).split())

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Going through normalisation

# Step 1 lowercasing
df['cleaned'] = df['texts'].apply(str.lower)

# Step 2 Some initial cleaning.
regex_to_clean = [r'(@.+?)\s', # Emails get captured! Usernames! Tags!
                  r'\s\d+\s',  # Digits! Numbers!
                  r'\s?\d+\.?', # Numbers not within spaces
                  r'(//t.co/.+?)\s', # Web domains with a space
                  r'(//t.co/.+?)'] # Web domains

# Get rid of all matches of the regex
for reg in regex_to_clean:
    df['cleaned'] = df['cleaned'].apply(regex_clean, regex=reg)

# Remove emojis
df['cleaned'] = df['cleaned'].apply(remove_emoji)

# Tokenise the clean version of the doucments

df['tokens'] = df['cleaned'].apply(word_tokenize)

# Create a list of stopwrods!

stpwrd = nltk.corpus.stopwords.words('english')

# Expand the list by adding punctuation in there!

# A lot of special characters
punc = '!"#$%&()*+, -./:;<=>?@[\]^_`{|}~”“\''

# Create list
punc = [x for x in punc]

# Extend stopword to extend list
stpwrd.extend(punc)

# Update tokens
df['tokens'] = df['tokens'].apply(lambda x:[words for words in x if words not in stpwrd])

# Remove tokens that are less than 3 characters

df['tokens'] = df['tokens'].apply(lambda document : [token for token in document if len(token)>2])

# Apply and remove the tokens

# Create a list of personal unwanted tokens

unwanted_words = ['remote','work']

df['tokens'] = df['tokens'].apply(lambda x:[words for words in x if words not in unwanted_words])


# Create a dictionary from the tokens provided

my_terms = Dictionary(documents = df['tokens'])

# Strat with empy dict
clean_dictionary = {}

for k,v in my_terms.cfs.items():
    if v>4: # If word appears more than 5 times
        # Add to clean dict with TF
        clean_dictionary[my_terms[k]] = v

print(df.head())
print(clean_dictionary)
