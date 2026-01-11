from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_USERNAME = "karthikeyan-datascientist"
REPO_NAME = "superkart-dataset"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
REPO_TYPE = "dataset"

DATA_FOLDER = "Files/data"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Check / create dataset repo
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Dataset repo '{REPO_ID}' created.")

# Upload dataset folder
api.upload_folder(
    folder_path=DATA_FOLDER,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print("✅ Dataset uploaded successfully to Hugging Face.")
