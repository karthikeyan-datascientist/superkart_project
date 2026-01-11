"""
Data Registration Script
Uploads raw Superkart dataset to Hugging Face Dataset Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# Configuration
# -----------------------------
HF_USERNAME = "karthikeyan-datascientist"
DATASET_NAME = "superkart-dataset"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"
REPO_TYPE = "dataset"

LOCAL_DATA_FOLDER = "Files/data"   # relative to GitHub repo

# -----------------------------
# Authenticate with HF
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Create dataset repo if needed
# -----------------------------
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Creating...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False
    )
    print(f"Dataset repo '{REPO_ID}' created.")

# -----------------------------
# Upload dataset folder
# -----------------------------
api.upload_folder(
    folder_path=LOCAL_DATA_FOLDER,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

print("✅ Dataset successfully registered on Hugging Face.")
