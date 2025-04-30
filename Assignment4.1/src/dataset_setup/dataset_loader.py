import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

 # define dataset class to wrap our image paths and labels
class ImageDataset(Dataset):
    # constructor for dataset class
    def __init__(self, paths, labels, transform):
        self.paths = paths      # list of file paths
        self.labels = labels    # list of image labels
        self.transform = transform  # transformations for each image

    # method to return number of images in dataset
    def __len__(self):
        return len(self.paths)  # total number of image paths in dataset

    # method to get image at a certain index
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')  # Load image with all 3 colors (RGB)
        
        # if transforms are defined
        if self.transform:
            image = self.transform(image)  # apply preprocessing transforms
        label = self.labels[idx]  # get corresponding label
        return image, label  # return the indexed sample (image, label)

# builds a list of (image_path, label tuples)
def build_dataset_list(dataset_dir, class_map):
    dataset = []

    # iterate through dataset and each subfolder
    for folder in sorted(os.listdir(dataset_dir)):
        label = class_map.get(folder)
        
        # skip folders that arent in the class map
        if label is None:
            continue

        folder_path = Path(dataset_dir) / folder
        
        # for each image, record the full path and its label
        for img_file in os.listdir(folder_path):
            dataset.append((str(folder_path / img_file), label))

    print(f"\nLoaded {len(dataset)} images from {Path(dataset_dir).name}\n") # output total image count
    return dataset

# convert img paths/labels into ImageDataset
def create_dataset(image_label_list, transform):
    # separate the list into individual lists of paths and labels
    paths, labels = zip(*image_label_list)

    # init the dataset with data
    ds = ImageDataset(paths, labels, transform)
    return ds 

# apply augmentation and minor preprocessing
def get_transforms(IMG_SIZE, DEVICE):
    print("Applying Augmentation to Training Data...")
    
    # training ds transforms
    train_transform = transforms.Compose([
    # augmentation
    transforms.RandomResizedCrop(IMG_SIZE),  # add random resizing
    transforms.RandomHorizontalFlip(), # randomly flip images horizontally
    transforms.RandomRotation(4), # rotate +/- 2 degrees randomly
    transforms.ColorJitter(brightness=0.1, contrast=0.2), # randomly change brightness/contrast

    # minor preprocessing
    transforms.ToTensor(), # convert to pytorch tensor
    transforms.Lambda(lambda x: x.to(DEVICE)) # force gpu usage
    ])

    # Val transforms/minor preprocessing
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE), # resize img properly
        transforms.ToTensor(), # convert to pytorch tensor
    ])
    return train_transform, val_transform