import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle


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

        # Save the model to pickle
        self.save_model()

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

        # Very similar to target column
        try:
            df.drop(columns=['work_interfere'], inplace = True)
        except:
            pass

        # Drop unrelated columns
        try:
            df.drop(columns=['mental_health_interview','phys_health_interview'], inplace = True)
        except:
            pass

        # Important feautures
        important_features = [
            "family_history",
            "obs_consequence",
            "Gender",
            "benefits",
            "care_options",
            "coworkers",
            "wellness_program",
            "seek_help",
            "no_employees",
            "mental_health_consequence",
            "mental_vs_physical",
            "supervisor",
            "anonymity",
            "Country",
            "Age"
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
            'n_estimators': [50, 100, 150],         # More trees = more learning
            'max_depth': [2, 3, 4, 5],                  # Controls tree complexity
            'learning_rate': [0.01, 0.05, 0.1],      # Lower = better generalisation (but needs more trees)
            'subsample': [0.8, 1.0],                 # Less than 1.0 = prevent overfitting
            'colsample_bytree': [0.8, 1.0],          # Use fewer features per tree
            'min_child_weight': [1, 5, 10],          # Larger = more conservative splits
            'gamma': [0, 1],                         # 0 = allow more splits, higher = only if split improves accuracy
        }


        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='mlogloss',
            random_state=42
        )

        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,  # Try only 10 random combos
            scoring='recall',
            cv=stratified_cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        # F1 micro
        # 0.79
        # 0.74
        grid.fit(self.X_train_fe, self.y_train)
        print("✅ Best XGBoost Params:", grid.best_params_)
        return grid.best_estimator_

    def evaluate_model(self):
        """Evaluate model using accuracy, classification report, and confusion matrix."""

        # Predict probabilities and convert to 0/1 using a 0.45 threshold
        y_train_probs = self.treeclf.predict_proba(self.X_train_fe)[:, 1]
        y_train_pred = (y_train_probs >= 0.401).astype(int)

        y_test_probs = self.treeclf.predict_proba(self.X_test_fe)[:, 1]
        y_test_pred = (y_test_probs >= 0.401).astype(int)

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

        for idx in sorted_idx[:15]:
            print(f"{features[idx]}: {importances[idx]:.4f}")

    def manual_check(self, y_test_pred):
        """Print the first 10 actual vs. predicted results for manual inspection."""
        df_final = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_test_pred})
        print(df_final.head(10))
        print("Dataframe length:" , len(df_final))

    def predict_from_model(self, input_df=None):
        """
        Accepts a DataFrame of one or more rows and returns binary predictions.
        Uses the saved encoders to preprocess input before prediction.
        """
        if input_df is None:
            # Example row
            input_df = pd.DataFrame({
                'family_history': ['No'],
                'obs_consequence': ['Yes'],
                "Gender": ["Male"],
                'benefits': ['No'],
                'care_options': ['No'],
                'coworkers': ['Yes'],
                'wellness_program': ['No'],
                'seek_help': ['No'],
                'no_employees': ['1-5'],
                'mental_health_consequence': ['Yes'],
                'mental_vs_physical': ['Yes'],
                'supervisor': ['No'],
                'anonymity': ['Yes'],
                "Country": ["United States"],
                "Age": [25],
            })

        # Preprocess using stored encoders
        input_fe = self.feature_engineering(input_df, training=False)

        # Predict
        prediction_proba = self.treeclf.predict_proba(input_fe)[:, 1]
        prediction_pred = (prediction_proba >= 0.6).astype(int)

        print("Prediction:", prediction_pred)
        return prediction_pred
    
    def save_model(self):
        # Save the model to a file
        with open("data/xgb_model.pkl", "wb") as f:
            pickle.dump(self.treeclf, f)


if __name__ == "__main__":
    ModelRemoteWorkerAnalysis().predict_from_model()
