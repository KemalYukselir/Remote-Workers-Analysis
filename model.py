from sklearn.ensemble import RandomForestClassifier
from eda import ClashEda
import numpy as np # Maths
import pandas as pd # General data use

from sklearn import metrics # Measure performance of DT model

# Import relevant DT libraries
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier # Import model
from sklearn import tree # DT Visuals
from sklearn.model_selection import train_test_split # Train test split
from sklearn.model_selection import GridSearchCV #
from sklearn.metrics import (confusion_matrix, accuracy_score) # Performance check
from sklearn.preprocessing import LabelEncoder # Encoding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# Load your dataset
df = ClashEda().get_dataframe()

class DecisionTreeModel():
    DEBUG = True
    def __init__(self):
        self.df_model = df.copy()
        # Encode target variable
        self.df_model['treatment'] = self.df_model['treatment'].map({'Yes': 1, 'No': 0})
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
        feature_cols.remove('treatment')

        X = self.df_model[feature_cols]
        y = self.df_model['treatment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        return X_train, X_test, y_train, y_test
    
    def check_target_balance(self):
        """
        - DT are sensitive to imbalance
        - Check and handel target class
        Results:
            treatment
            1    0.506044
            0    0.493956
        """
        print(self.df_model['treatment'].value_counts(normalize=True))

    def feature_engineering(self, df):
        ##################
        ## Feature engineering
        ##################

        df = df.copy()
        
        important_features = [
            "work_interfere",
            "family_history",
            "care_options",
            "benefits",
            "anonymity",
            "coworkers",
            "phys_health_interview",
            "obs_consequence",
            "Country",
            "mental_health_consequence"
        ]
        df = df[important_features]

        le = LabelEncoder()
        for col in df.select_dtypes(include='object'):
            df[col] = le.fit_transform(df[col])

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
        Fit a multi-class XGBoost model using GridSearchCV with SMOTE for class imbalance
        """

        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Setup XGBoost for multi-class classification
        model = XGBClassifier(
            objective='binary:logistic',   # returns probability distributions
            eval_metric='mlogloss',
            random_state=42
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            verbose=0,   # 👈 silent!
            n_jobs=-1
        )

        # Fit the model
        grid.fit(self.X_train_fe, self.y_train)

        print("✅ Best XGBoost Params:", grid.best_params_)
        return grid.best_estimator_


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

        y_test_probs = self.treeclf.predict_proba(self.X_test_fe)[:, 1]
        y_test_pred = (y_test_probs >= 0.7).astype(int)  # lower threshold to improve recall

        print("\n📊 Classification Report (threshold=0.4):")
        print(classification_report(self.y_test, y_test_pred))


    def show_feature_importance(self):
        """ Show the top coloumns with the most significance"""
        importances = self.treeclf.feature_importances_
        features = self.X_train_fe.columns
        sorted_idx = np.argsort(importances)[::-1]

        for idx in sorted_idx[:10]:  # top 10
            print(f"{features[idx]}: {importances[idx]:.4f}")


    def manual_check(self, y_test_pred):
        """ Eyeball check of the predictions """
        # Manually checking predictions
        df_final = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_test_pred})
        print(df_final.head(10))


if __name__ == "__main__":
    DecisionTreeModel()