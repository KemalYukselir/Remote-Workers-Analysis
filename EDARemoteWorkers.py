import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EDARemoteWorkers():
    def __init__(self):
        # Import CSV
        self.df = pd.read_csv('data/survey.csv')
        self.df_clean = self.df.copy()
        self.drop_unused_columns()
        self.handle_nulls()
        self.map_gender_values()
        self.check_unique()
        self.summary_statistics()

    def drop_unused_columns(self):
        """ Cols that provide no value """
        cols_to_drop = ["comments","Timestamp","state"]
        self.df_clean = self.df_clean.drop(columns=cols_to_drop)

    def handle_nulls(self):
        """
        Check and handle nulls
        - Only one row with null - can be dropped
        """
        self.df_clean.info()
        print(self.df_clean.isnull().sum())

        # Target - Have to ha nulls
        self.df_clean = self.df_clean.dropna(subset=["self_employed"])
        self.df_clean["work_interfere"] = self.df_clean["work_interfere"].fillna("X")
        print(self.df_clean.isnull().sum())
    
    def map_gender_values(self):
        """
        Check for inconsistent values in the dataset.
        """
        # Define gender groups
        male_list = [
            'male', 'Male', 'M', 'm', 'Cis Male', 'Male (CIS)', 'cis male', 'Male ', 'Man', 'Cis Man',
            'Mal', 'Malr', 'maile', 'Make', 'msle', 'Mail', 'ostensibly male, unsure what that really means'
        ]

        female_list = [
            'female', 'Female', 'F', 'f', 'Cis Female', 'Woman', 'woman', 'Femake', 'Female ', 'Female (cis)',
            'femail', 'cis-female/femme'
        ]

        nonbinary_list = [
            'Trans-female', 'Trans woman', 'Agender', 'Androgyne', 'Genderqueer', 'non-binary',
            'queer', 'fluid', 'queer/she/they', 'Enby', 'Guy (-ish) ^_^', 'something kinda male?',
            'male leaning androgynous'
        ]

        unknown_list = [
            'Nah', 'All', 'A little about you', 'p'
        ]

        # Function to clean gender values
        def clean_gender(gender):
            if pd.isna(gender):
                return 'Other/Unspecified'

            gender = gender.strip()

            if gender in male_list:
                return 'Male'
            elif gender in female_list:
                return 'Female'
            elif gender in nonbinary_list:
                return 'Non-binary'
            elif gender in unknown_list:
                return 'Other/Unspecified'
            else:
                return 'Other/Unspecified'

        # Apply the function to your dataset
        self.df_clean['Gender'] = self.df_clean['Gender'].apply(clean_gender)
    
    def check_unique(self):
        """ 
        Print all unique values in each column of the dataset.
        - Decide what to do with every column.
        """
        for col in self.df_clean.columns:
            print(f"\n{col}: {self.df_clean[col].unique()}\n")

    def summary_statistics(self):
        """ Check numerical stats of every column """
        print(self.df_clean.shape)
        print(self.df_clean.describe())
        print(self.df_clean.info())
        print(self.df_clean['remote_work'].value_counts())

    def get_dataframe(self):
        return self.df_clean
    
    def save_dataframe(self, filename):
        """ Save the dataframe to a CSV file """
        """ Save only remote workers"""
        self.df_clean = self.df_clean[self.df_clean['remote_work'] == 'Yes']
        self.df_clean.to_csv(filename, index=False)
        print(f"Dataframe saved to {filename}")

if __name__ == "__main__":
    EDARemoteWorkers()