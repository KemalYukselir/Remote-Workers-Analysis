from eda import ClashEda
import numpy as np # Maths
import pandas as pd # General data use

from sklearn import metrics # Measure performance of DT model

# Import relevant DT libraries
from sklearn.tree import DecisionTreeClassifier # Import model
from sklearn import tree # DT Visuals
from sklearn.model_selection import train_test_split # Train test split
from sklearn.model_selection import GridSearchCV #
from sklearn.metrics import (confusion_matrix, accuracy_score) # Performance check
from sklearn.preprocessing import LabelEncoder # Encoding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your dataset
df = ClashEda().get_dataframe()

class DecisionTreeModel():
    DEBUG = True
    def __init__(self):
        self.df_model = df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        self.check_target_balance()
        # self.treeclf = self.fit_model()
        # self.evaluate_model()

    def split_train_test(self):
        """ 
        - Set target and features 
        - Split train and test to 80% 20%
        - Check for no index errors
        """
        feature_cols = list(self.df_model.columns)

        # Target is Mental_Health_Condition
        feature_cols.remove('Mental_Health_Condition')
        X = self.df_model[feature_cols]
        y = self.df_model['Mental_Health_Condition']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4992)

        return X_train, X_test, y_train, y_test
    
    def check_target_balance(self):
        """
        - DT are sensitive to imbalance
        - Check and handel target class
        Results:
            - Burnout             0.2560
            - Anxiety             0.2556
            - Depression          0.2492
            - No Mental Health    0.2392
        """
        print(self.df_model['Mental_Health_Condition'].value_counts(normalize=True))

    def fit_model(self):
        """ 
        - Fit train to Decision Tree 
        """
        # Creating model with fine tune parqams (Max depth 4 from grid search)
        treeclf = DecisionTreeClassifier(max_depth=4)

        # Fitting/training model with the data
        treeclf.fit(self.X_train, self.y_train)

        return treeclf
    
    def evaluate_model(self):
        """ 
        - Generate y predictions
        - Evaluate on accuracy, precision, recall, f1
        """

        y_pred = self.treeclf.predict(self.X_test) # Getting predictions

        # Check metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    DecisionTreeModel()