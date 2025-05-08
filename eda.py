import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ClashEda():
    DEBUG = True
    def __init__(self):
        # Import CSV
        self.df = pd.read_csv('data/survey.csv')
        self.df_clean = self.df.copy()
        self.drop_unused_columns()
        self.handle_nulls()
        self.check_unique()
        # if self.DEBUG:
        #     self.run()
    
    def check_unique(self):
        """ 
        Print all unique values in each column of the dataset.
        - Decide what to do with every column.
        """
        for col in self.df_clean.columns:
            print(f"\n{col}: {self.df_clean[col].unique()}\n")
    
    def handle_nulls(self):
        """
        Check and handle nulls
        - Only one row with null - can be dropped
        """
        if self.DEBUG:
            self.df_clean.info()
            print(self.df_clean.isnull().sum())

        # Target - Have to ha nulls
        self.df_clean = self.df_clean.dropna(subset=["self_employed"])
        self.df_clean["work_interfere"] = self.df_clean["work_interfere"].fillna("X")
        if self.DEBUG:
            print(self.df_clean.isnull().sum())

    def drop_unused_columns(self):
        cols_to_drop = ["comments","Timestamp","state"]
        self.df_clean = self.df_clean.drop(columns=cols_to_drop)
    
    def summary_statistics(self):
        """ Check numerical stats of every column """
        print(self.df_clean.shape)
        print(self.df_clean.describe())
        print(self.df_clean.info())

    def correlation_matrix(self):
        """
        Display a zoomed-out correlation matrix to better fit the screen.
        """

        # Convert target to numeric if not already
        # self.df_clean['Mental_Health_Condition'] = self.df_clean['Mental_Health_Condition'].astype('category').cat.codes

        plt.figure(figsize=(14, 8))  # Wider and taller figure
        correlation = self.df_clean.corr(numeric_only=True)

        sns.heatmap(
            correlation,
            annot=True,
            cmap='coolwarm',
            fmt=".3f",
            annot_kws={"size": 8},  # smaller font for annotations
            cbar_kws={"shrink": 0.75}  # smaller colorbar
        )

        plt.title("Correlation Matrix with Target", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()

    def run(self):
        """ Auto run when class is called """
        self.summary_statistics()
        self.correlation_matrix()
    
    def get_dataframe(self):
        return self.df_clean

if __name__ == "__main__":
    ClashEda()