import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class ModelRemoteWorkerAnalysis():
    DEBUG = True

    def __init__(self):
        # Dictionary to store encoders for categorical columns
        self.encoders = {}

        # Load the dataset
        self.df_model = pd.read_csv('data/remote_workers_clean.csv')

        # Encode target column ('Yes' -> 1, 'No' -> 0)
        self.df_model['treatment'] = self.df_model['treatment'].map({'Yes': 1, 'No': 0})

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()

        # Print balance of target values
        self.check_target_balance()

        # Feature engineering + encoding (fit encoders on training set only)
        self.X_train_fe = self.feature_engineering(self.X_train, training=True)
        self.X_test_fe = self.feature_engineering(self.X_test, training=False)

        # Train the XGBoost model using grid search
        self.treeclf = self.fit_model()

        # Evaluate the model
        self.evaluate_model()

    def split_train_test(self):
        """Split the dataset into training and test sets (80/20)."""
        feature_cols = list(self.df_model.columns)
        feature_cols.remove('treatment')  # Remove target column

        X = self.df_model[feature_cols]
        y = self.df_model['treatment']

        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def check_target_balance(self):
        """Print the percentage of each class in the target column."""
        print(self.df_model['treatment'].value_counts(normalize=True))

    def feature_engineering(self, df, training=True):
        """
        Select important features and apply consistent label encoding.
        During training, encoders are fit and stored.
        During prediction, the same encoders are reused.
        """
        df = df.copy()

        # Manually selected relevant features
        important_features = [
            "work_interfere", "family_history", "care_options", "benefits", "anonymity",
            "coworkers", "phys_health_interview", "obs_consequence", "Country",
            "mental_health_consequence"
        ]

        df = df[important_features]

        for col in df.select_dtypes(include='object'):
            if training:
                # Fit encoder on training data
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le  # Store the encoder
            else:
                # Reuse saved encoder for prediction
                le = self.encoders.get(col)
                if le:
                    # Safely handle unseen values by assigning -1
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                else:
                    raise ValueError(f"No encoder found for column: {col}")
        return df

    def fit_model(self):
        """Train XGBoost using GridSearchCV and return the best model."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='mlogloss',
            random_state=42
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            verbose=0,
            n_jobs=-1
        )

        grid.fit(self.X_train_fe, self.y_train)
        print("✅ Best XGBoost Params:", grid.best_params_)
        return grid.best_estimator_

    def evaluate_model(self):
        """Evaluate model using accuracy, classification report, and confusion matrix."""

        # Predict probabilities and convert to 0/1 using a 0.45 threshold
        y_train_probs = self.treeclf.predict_proba(self.X_train_fe)[:, 1]
        y_train_pred = (y_train_probs >= 0.45).astype(int)

        y_test_probs = self.treeclf.predict_proba(self.X_test_fe)[:, 1]
        y_test_pred = (y_test_probs >= 0.45).astype(int)

        # Show metrics
        print(f"\n🧪 Training Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")
        print(f"🧾 Testing Accuracy:  {accuracy_score(self.y_test, y_test_pred):.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_test_pred))
        print("\n📉 Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_test_pred))

        # Manually inspect predictions
        self.manual_check(y_test_pred)

        print("\nMost important features:\n")
        self.show_feature_importance()

    def show_feature_importance(self):
        """Print top 10 features based on importance from XGBoost."""
        importances = self.treeclf.feature_importances_
        features = self.X_train_fe.columns
        sorted_idx = np.argsort(importances)[::-1]

        for idx in sorted_idx[:10]:
            print(f"{features[idx]}: {importances[idx]:.4f}")

    def manual_check(self, y_test_pred):
        """Print the first 10 actual vs. predicted results for manual inspection."""
        df_final = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_test_pred})
        print(df_final.head(10))

    def predict_from_model(self, input_df=None):
        """
        Accepts a DataFrame of one or more rows and returns binary predictions.
        Uses the saved encoders to preprocess input before prediction.
        """
        if input_df is None:
            # Example row
            input_df = pd.DataFrame({
                'work_interfere': ['Sometimes'],
                'family_history': ['Yes'],
                'care_options': ['No'],
                'benefits': ['No'],
                'anonymity': ['Yes'],
                'coworkers': ['Yes'],
                'phys_health_interview': ['Yes'],
                'obs_consequence': ['Yes'],
                'Country': ['United States'],
                'mental_health_consequence': ['Yes']
            })

        # Preprocess using stored encoders
        input_fe = self.feature_engineering(input_df, training=False)

        # Predict
        prediction_proba = self.treeclf.predict_proba(input_fe)[:, 1]
        prediction_pred = (prediction_proba >= 0.6).astype(int)

        print("Prediction:", prediction_pred)
        return prediction_pred


if __name__ == "__main__":
    ModelRemoteWorkerAnalysis().predict_from_model()
