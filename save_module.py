import joblib
from ModelRemoteWorkerAnalysis import ModelRemoteWorkerAnalysis

m = ModelRemoteWorkerAnalysis()
joblib.dump(m, 'data/xgb_model.pkl')  # Save only the model