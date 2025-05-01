# import os
# import zipfile
# from kaggle.api.kaggle_api_extended import KaggleApi

# IMPLEMENT IF TIME ALLOWS
# function to retrieve dataset from kaggle
# def download_and_extract_urbansound8k(destination_dir="./Assignment4.1/dataset"):
#     dataset_slug = "urbansound8k"  # actual Kaggle dataset is hosted under an external source
#     dataset_owner = "chrisfilo"   # current known uploader on Kaggle

#     dataset_name = f"{dataset_owner}/{dataset_slug}"
#     zip_path = os.path.join(destination_dir, "UrbanSound8K.zip")

#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)

#     # Use Kaggle API
#     api = KaggleApi()
#     api.authenticate()

#     print("Downloading UrbanSound8K...")
#     api.dataset_download_files(dataset_name, path=destination_dir, unzip=False)

#     print("Extracting UrbanSound8K...")
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(destination_dir)

#     os.remove(zip_path)  # cleanup
#     print("UrbanSound8K is ready at:", destination_dir)
