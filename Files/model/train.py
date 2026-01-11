import pandas as pd
import joblib
import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Hugging Face setup
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "karthikeyan-datascientist/superkart-dataset"
MODEL_REPO = "karthikeyan-datascientist/superkart-sales-model"

api = HfApi(token=HF_TOKEN)

# -----------------------------
# Load prepared data
# -----------------------------
Xtrain = pd.read_csv(
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/Xtrain.csv"
)
Xtest = pd.read_csv(
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/Xtest.csv"
)
ytrain = pd.read_csv(
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/ytrain.csv"
).squeeze()
ytest = pd.read_csv(
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/ytest.csv"
).squeeze()

print("✅ Training data loaded")

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(Xtrain, ytrain)

# -----------------------------
# Evaluate
# -----------------------------
train_pred = model.predict(Xtrain)
test_pred = model.predict(Xtest)

print("Train R2:", r2_score(ytrain, train_pred))
print("Test R2:", r2_score(ytest, test_pred))
rmse = mean_squared_error(ytest, test_pred) ** 0.5
print("Test RMSE:", rmse)

# -----------------------------
# Save model
# -----------------------------
MODEL_FILE = "sales_model.joblib"
joblib.dump(model, MODEL_FILE)

# -----------------------------
# Create / Upload model repo
# -----------------------------
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print("Model repo already exists")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print("Model repo created")

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model",
)

print("✅ Model trained and uploaded to Hugging Face")
