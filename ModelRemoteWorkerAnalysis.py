import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class ModelRemoteWorkerAnalysis():
    DEBUG = True

    def __init__(self):
        # Load the dataset
        self.df_model = pd.read_csv('data/remote_workers_clean.csv')

        self.label_encoders = {}

        # Format for model
        self.format_Dataframe()

        # Encode target column ('Yes' -> 1, 'No' -> 0)
        self.df_model['treatment'] = self.df_model['treatment'].map({'Yes': 1, 'No': 0, 1:1, 0:0})

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()

        # Print balance of target values
        self.check_target_balance()

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

    def format_Dataframe(self, predict_df=None):
        drop_cols = [
            'mental_health_interview', 'phys_health_interview', 'tech_company', "self_employed",
            'Country', 'Gender', 'Age', "remote_work", "phys_health_consequence"
        ]

        if predict_df is None:
            # Training mode
            self.df_model.drop(columns=drop_cols, errors='ignore', inplace=True)

            if 'work_interfere' in self.df_model.columns:
                self.df_model['work_interfere'] = self.df_model['work_interfere'].replace({
                    'Often': 'Yes',
                    'Sometimes': 'Yes',
                    'Rarely': 'No',
                    'Never': 'No',
                    'X': 'No'
                })


            categorical_cols = self.df_model.select_dtypes(include='object').columns
            for col in categorical_cols:
                le = LabelEncoder()
                self.df_model[col] = le.fit_transform(self.df_model[col])
                self.label_encoders[col] = le  # Store encoder

        else:
            # Prediction mode
            predict_df.drop(columns=drop_cols, errors='ignore', inplace=True)

            if 'work_interfere' in self.df_model.columns:
                self.df_model['work_interfere'] = self.df_model['work_interfere'].replace({
                    'Often': 'Yes',
                    'Sometimes': 'Yes',
                    'Rarely': 'No',
                    'Never': 'No',
                    'X': 'No'
                })

            for col in predict_df.select_dtypes(include='object').columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    try:
                        predict_df[col] = le.transform(predict_df[col])
                    except ValueError:
                        # Handle unseen categories (map to -1 or most common value)
                        predict_df[col] = predict_df[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                else:
                    # Column wasn't seen during training — fill with 0 or remove
                    predict_df[col] = 0

            return predict_df
       

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

        stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,  # Try only 50 random combos
            # scoring='recall_micro',
            cv=stratified_cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        grid.fit(self.X_train, self.y_train)
        print("✅ Best XGBoost Params:", grid.best_params_)
        return grid.best_estimator_

    def evaluate_model(self):
        """Evaluate model using accuracy, classification report, and confusion matrix."""

        # Predict probabilities and convert to 0/1 using a 0.45 threshold
        y_train_probs = self.treeclf.predict_proba(self.X_train)[:, 1]
        y_train_pred = (y_train_probs >= 0.52).astype(int)

        y_test_probs = self.treeclf.predict_proba(self.X_test)[:, 1]
        y_test_pred = (y_test_probs >= 0.52).astype(int)

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
        features = self.X_train.columns
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
        Run a prediction on a predefined example input.
        Returns:
            int: Predicted class (0 = No treatment, 1 = Needs treatment)
        """
        if input_df is None:
            input_df = pd.DataFrame({
                'work_interfere': ['X'], # If you have a mental health condition, could it pontentially interfere with your work?
                'family_history': ['Yes'],  # Do you have a family history of mental illness?
                'obs_consequence': ['No'],  # Observed consequences for coworkers?
                'benefits': ['Yes'],  # Employer provides mental health benefits?
                'care_options': ['Yes'],  # Aware of care options?
                'coworkers': ['Yes'],  # Talk to coworkers about mental health?
                'wellness_program': ['Yes'],  # Employer discussed mental health?
                'seek_help': ['Yes'],  # Resources to seek help?
                'no_employees': ['100-500'],  # Company size
                'mental_health_consequence': ['Yes'],  # Negative consequence of disclosure?
                'mental_vs_physical': ['Yes'],  # Mental vs physical health seriousness
                'supervisor': ['Yes'],  # Talk to supervisor?
                'anonymity': ['Yes'],  # Is anonymity protected?
                'leave': ['Very easy']
            })

        processed = self.format_Dataframe(input_df)

        # Align the column order with training data
        processed = processed[self.X_train.columns]

        # Predict
        threshold=0.52
        probs = self.treeclf.predict_proba(processed)[:, 1]
        pred = int(probs[0] >= threshold)

        print(f"\n🧠 Prediction: {pred} (1 = Needs treatment, 0 = Doesn’t)")
        print(f"🔍 Probability: {probs[0]:.2}")

        return pred


if __name__ == "__main__":
    model = ModelRemoteWorkerAnalysis()
    model.predict_from_model()
