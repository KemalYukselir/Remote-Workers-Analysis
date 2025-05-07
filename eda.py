import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ClashEda():
    DEBUG = True
    def __init__(self):
        # Import CSV
        self.df = pd.read_csv('data/Impact_of_Remote_Work_on_Mental_Health.csv')
        self.df_clean = ""
        self.handle_nulls()
        if self.DEBUG:
            self.run()
    
    def check_unique(self):
        """ 
        Print all unique values in each column of the dataset.
        - Decide what to do with every column.
        """
        for col in self.df.columns:
            print(f"\n{col}: {self.df[col].unique()}\n")
    
    def handle_nulls(self):
        """
        Check and handle nulls
        - Only one row with null - can be dropped
        """
        self.df_clean = self.df.copy()
        if self.DEBUG:
            self.df_clean.info()
            print(self.df_clean.isnull().sum())

        # Target - Have to drop nulls
        self.df_clean.dropna(subset=["Mental_Health_Condition"],inplace=True)
        self.df_clean["Physical_Activity"] = self.df_clean["Physical_Activity"].fillna("Unknown")
        if self.DEBUG:
            print(self.df_clean.isnull().sum())
    
    def summary_statistics(self):
        """ Check numerical stats of every column """
        print(self.df_clean.shape)
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
        self.summary_statistics()
        self.correlation_matrix()
    
    def get_dataframe(self):
        return self.df_clean

if __name__ == "__main__":
    ClashEda()