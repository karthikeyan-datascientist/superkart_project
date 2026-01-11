from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

SPACE_REPO = "karthikeyan-datascientist/superkart-project"

api.upload_folder(
    folder_path="Files/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
)

print("✅ Application deployed to Hugging Face Space")
