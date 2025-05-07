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

    def split_train_test(self):
        """ 
        - Set target and features 
        - Split train and test to 80% 20%
        - Check for no index errors
        """
        feature_cols = list(self.df_model.columns)
        feature_cols.remove('Mental_Health_Condition')
        X = self.df_model[feature_cols]
        y = self.df_model['Mental_Health_Condition']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=4992)

        # Check for no index errors
        if self.DEBUG:
            print(all(self.X_train.index == self.y_train.index))
            print(all(self.X_test.index == self.y_test.index))