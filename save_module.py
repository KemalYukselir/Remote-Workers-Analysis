import joblib
from ModelRemoteWorkerAnalysis import ModelRemoteWorkerAnalysis

m = ModelRemoteWorkerAnalysis()
joblib.dump(m.treeclf, 'data/xgb_model.pkl')  # Save only the model