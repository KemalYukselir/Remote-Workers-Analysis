import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ClashEda():

    def __init__(self):
        # Import CSV
        self.df = pd.read_csv('data/clash_royale_cards.csv')
        self.df_clean = ""

        self.run()
    
    def check_unique(self):
        """ 
        Print all unique values in each column of the dataset.
        - Decide what to do with every column.
        """
        for col in self.df.columns:
            print(f"{col}: {self.df[col].unique()}")
    
    def handle_nulls(self):
        """
        Check and handle nulls
        - Only one row with null - can be dropped
        """
        self.df_clean = self.df.copy()
        self.df_clean.info()
        print(self.df_clean.isnull().sum())

        # Only one null -> Can be dropped
        self.df_clean.dropna(inplace=True)
        print(self.df_clean.isnull().sum())
    
    def summary_statistics(self):
        """ Check numerical stats of every column """
        print(self.df_clean.describe())

    def correlation_matrix(self):
        """ Check correlation between every feature """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df_clean.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    def run(self):
        """ Auto run when class is called """
        self.check_unique()
        self.handle_nulls()
        self.summary_statistics()
        self.correlation_matrix()

if __name__ == "__main__":
    ClashEda()