import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter

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

        print(f"\nLoaded {len(dataset)} images.") # output total image count
    return dataset

# function to create pytorch dataloaders from list of image path's and their labels
def create_dataset(image_label_list, img_size, batch_size, num_workers, class_weights):
    # separate the list into individual lists of paths and labels
    paths, labels = zip(*image_label_list)

    # Apply transformations as preprocessing
    transform = transforms.Compose([
        transforms.Resize(img_size),  # resize all images to defined img_size
        transforms.ToTensor(),        # convert to pytorch tensor
    ])

    # init the dataset with data
    ds = ImageDataset(paths, labels, transform)
    
    sampler = None
    if class_weights:
        # generate weights for each sample
        sample_weights = [class_weights[label] for _, label in image_label_list]
        
        # create weighted sampler
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

    # create pytorch dataloader
    loader = DataLoader(
        ds,
        batch_size=batch_size,  # images per batch
        sampler=sampler, # apply weights for class distribution
        num_workers=num_workers,  # how many cpu processes to use
        pin_memory=True          # use cuda pin memory
    )
    return loader  # return the created dataLoader

# function to determine class weights, allows us to use all data without bias
def get_class_weights(image_labels_list, num_classes):
    # count how many images/labels per class
    label_counts = Counter(label for _, label in image_labels_list)

    # toal number of images in dataset
    total_images = sum(label_counts.values())

    # get the class weights
    class_weights = {
        class_id: total_images / (num_classes * count)
        for class_id, count in label_counts.items()
    }
    
     # output the class weights
    print("\n---------- Class Weights ----------\n")
    for class_id, weight in class_weights.items():
        print(f"Class {class_id}: {weight:.4f}")
    print("\n------------------------------------\n")
    
    return class_weights

# function to return the pytorch dataloader objects 
def get_dataloaders(train_list, val_list, num_classes, img_size, batch_size, num_workers):
    # get class weights for training set
    train_weights = get_class_weights(train_list, num_classes)

    # create training dataset w/ shuffled data and weighted samples
    train_loader = create_dataset(
        train_list, img_size, batch_size,
        num_workers=num_workers,
        class_weights=train_weights
    )

    # create validation dataset with no shuffling or weighted samples
    val_loader = create_dataset(
        val_list, img_size, batch_size,
        num_workers=num_workers,
        class_weights=None
    )

    return train_loader, val_loader # return datasets
