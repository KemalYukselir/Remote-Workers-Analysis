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


    def get_comments_common_words(self):
        """Generate a bar plot to visualize the most common words in comments."""
        # Extract comments
        word_bank = ""
        # Comments with no NaN
        comments = self.df.loc[self.df['comments'].notna(), 'comments']
        for comment in comments:
            word_bank += str(comment) + " "

        word_bank = word_bank.lower().replace("(","").replace(")","").split(" ")
        print(word_bank)
        lemmatizer = WordNetLemmatizer() # Lemmatizer tool
        p_stemmer = PorterStemmer() # Stemmer tool
        s_stemmer = SnowballStemmer(language='english') # Stemmer tool

        stpwrd = nltk.corpus.stopwords.words('english') # Stop words
        stpwrd.extend(string.punctuation)

        # Add custom stop words
        manual_stpwrd = ['–','','wa','ha','health','employer','would','get','remote','would','get']
        stpwrd.extend(manual_stpwrd)
        
        lemma = [lemmatizer.lemmatize(x) for x in word_bank]
        porter = [p_stemmer.stem(x) for x in word_bank]
        snowball = [s_stemmer.stem(x) for x in word_bank]
        print(list(zip(word_bank, lemma)))
        
        lemma = [x for x in lemma if x not in stpwrd]
        porter = [x for x in porter if x not in stpwrd]
        snowball = [x for x in snowball if x not in stpwrd]
        # Count top words
        word_count_lemma = Counter(lemma)
        # word_count_porter = Counter(porter)
        # word_count_snowball = Counter(snowball)


        # Visual with barplot
        word_count_lemma_df = pd.DataFrame(word_count_lemma.most_common(10), columns=['Word', 'Count'])
        # word_count_porter_df = pd.DataFrame(word_count_porter.most_common(10), columns=['Word', 'Count'])
        # word_count_snowball_df = pd.DataFrame(word_count_snowball.most_common(10), columns=['Word', 'Count'])

        print(word_count_lemma_df.head(10))
        # print(word_count_porter_df.head(10))
        # print(word_count_snowball_df.head(10))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Word', y='Count', data=word_count_lemma_df, palette='viridis', ax=ax)
        ax.set_title('Top 10 Common Words in comments (Remote Workers)', fontsize=16)
        ax.set_xlabel('Word', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)

        # Show figure
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    nlp = NLPMentalHealth()
    nlp.get_comments_common_words()