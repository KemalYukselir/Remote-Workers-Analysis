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
from sklearn.metrics import classification_report

# Load your dataset
df = ClashEda().get_dataframe()

class DecisionTreeModel():
    DEBUG = True
    def __init__(self):
        self.df_model = df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        self.check_target_balance()
        self.X_train_fe , self.X_test_fe = self.prepare_features()
        self.treeclf = self.fit_model()
        self.evaluate_model()

    def split_train_test(self):
        """ 
        - Set target and features 
        - Split train and test to 80% - 20%
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

    def feature_engineering(self, df):
        ##################
        ## Feature engineering
        ##################

        df = df.copy()

        # Drop irrelevant or highly collinear columns
        df.drop(columns=[
            "Employee_ID",  # Useless
        ], inplace=True)

        df = pd.get_dummies(df, drop_first=True)

        # for col in df.select_dtypes(include='object'):
        #     le = LabelEncoder()
        #     df[col] = le.fit_transform(df[col])

        return df

    def prepare_features(self):
        ##################
        ## Feature engineering
        ##################

        print(self.df_model.info())

        for col in self.df_model.select_dtypes(include='object').columns:
            print(f"\n{col}: {self.df_model[col].unique()}\n")

        # Feature engineering for train and test sets
        X_train_fe = self.feature_engineering(self.X_train)
        X_test_fe = self.feature_engineering(self.X_test)

        # Check index
        if self.DEBUG:
            print(all(X_train_fe.index == self.X_train.index))
            print(all(X_test_fe.index == self.X_test.index))

        return X_train_fe, X_test_fe

    def fit_model(self):
        """ 
        - Fit and fine-tune a Decision Tree using GridSearchCV
        """
        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(), 
            param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],         # Splitting criteria
                'max_depth': [3, 5, 7, 10, None],                    # Tree depth control
                'min_samples_split': [2, 5, 10],                     # Minimum samples to split an internal node
                'min_samples_leaf': [1, 2, 4],                       # Minimum samples required at a leaf node
            },            
            cv=10,
            refit=True,
            verbose=2,
            scoring='accuracy'  # Use 'recall_weighted' for multiclass support
        )
        
        grid.fit(self.X_train_fe, self.y_train)

        # Use the best model found
        treeclf = grid.best_estimator_

        return treeclf
    
    def evaluate_model(self):
        """ 
        Evaluate model on accuracy, precision, recall, F1, and confusion matrix
        """
        y_train_pred = self.treeclf.predict(self.X_train_fe)
        y_test_pred = self.treeclf.predict(self.X_test_fe)

        print(f"\n🧪 Training Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")
        print(f"🧾 Testing Accuracy:  {accuracy_score(self.y_test, y_test_pred):.4f}")

        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_test_pred))

        print("\n📉 Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_test_pred))

        self.manual_check(y_test_pred)

        print("\nMost important features\n")

        self.show_feature_importance()

    def show_feature_importance(self):
        importances = self.treeclf.feature_importances_
        features = self.X_train_fe.columns
        sorted_idx = np.argsort(importances)[::-1]
        print(sorted_idx)


    def manual_check(self, y_test_pred):
        # Manually checking predictions
        df_final = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_test_pred})
        print(df_final.tail(10))


if __name__ == "__main__":
    DecisionTreeModel()