from ModelRemoteWorkerAnalysis import ModelRemoteWorkerAnalysis
import pickle


# Train and save model
model = ModelRemoteWorkerAnalysis()

with open("data/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)