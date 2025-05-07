from eda import ClashEda
import numpy as np # Maths
import pandas as pd # General data use

from sklearn import metrics # Measure performance of DT model

# Import relevant DT libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree # Visuals

# Load your dataset
df = ClashEda().get_dataframe()

class DecisionTreeModel():
    def __init__(self):
        self.df_model = df.copy()
    