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