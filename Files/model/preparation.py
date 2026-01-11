import pandas as pd
from huggingface_hub import HfApi
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Hugging Face setup
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO = "karthikeyan-datascientist/superkart-dataset"

# Load dataset from HF
df = pd.read_csv(
    "https://huggingface.co/datasets/karthikeyan-datascientist/superkart-dataset/resolve/main/SuperKart.csv"
)

print("Dataset loaded successfully")

# Drop ID columns
df.drop(columns=["Product_Id", "Store_Id"], inplace=True)

# Encode categorical columns
categorical_cols = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split
X = df.drop("Product_Store_Sales_Total", axis=1)
y = df["Product_Store_Sales_Total"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload splits to HF
for file in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

print("✅ Data preparation completed and uploaded")
