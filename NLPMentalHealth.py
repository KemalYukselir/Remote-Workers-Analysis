# Visual
from matplotlib import pyplot as plt
import seaborn as sns 
import pandas as pd # Data manipulation

# Helper tools
from collections import Counter # Counting things
import string # Contains String stuffs

# NLTK suite
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# Download list of words and characters
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class NLPMentalHealth:
    def __init__(self):
        self.df = pd.read_csv('data/survey.csv')
        # Only remote workers and those are burn out (treament)
        self.df = self.df[self.df["remote_work"] == "Yes"]
        print(self.df.shape)
        self.df = self.df[self.df["treatment"] == "Yes"]
        print(self.df.shape)
        self.df = self.df[self.df["self_employed"] == "No"]
        print(self.df.shape)

        self.words_dictionary = self.get_comments_common_words()
        self.world_cloud()


    def get_comments_common_words(self):
        """Generate a bar plot to visualize the most common words in comments."""
        # Extract comments
        word_bank = ""
        # Comments with no NaN
        comments = self.df.loc[self.df['comments'].notna(), 'comments']
        for comment in comments:
            word_bank += str(comment) + " "

        word_bank = word_bank.lower().replace("(","").replace(")","").split(" ")
        lemmatizer = WordNetLemmatizer() # Lemmatizer tool

        stpwrd = nltk.corpus.stopwords.words('english') # Stop words
        stpwrd.extend(string.punctuation)

        # Add custom stop words
        manual_stpwrd = ['–','','wa','ha','health','employer','would','get','remote','would','get']
        stpwrd.extend(manual_stpwrd)
        
        lemma = [lemmatizer.lemmatize(x) for x in word_bank]
        print(list(zip(word_bank, lemma)))
        
        lemma = [x for x in lemma if x not in stpwrd]
        # Count top words
        word_count_lemma = Counter(lemma)

        # Visual with barplot
        word_count_lemma_df = pd.DataFrame(word_count_lemma.most_common(10), columns=['Word', 'Count'])

        # Return dict
        words_dictionary = dict(word_count_lemma_df.values)

        return words_dictionary




if __name__ == "__main__":
    nlp = NLPMentalHealth()