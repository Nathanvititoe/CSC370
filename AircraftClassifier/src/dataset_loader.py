# for image loading & preprocessing
from torchvision import datasets, transforms  # type: ignore 

# function to load and prepare the dataset for the model
def load_dataset(data_dir, img_size):
    transform = transforms.Compose([
        # Augmentation
        transforms.Resize(img_size), # resize images to match img_size resolution
        transforms.RandomHorizontalFlip(p=0.6), # random flips for augmentation (prevent overfit)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # adjust brightness/contrast for augmentation
        transforms.RandomRotation(10),
        transforms.ToTensor(), # convert from [255,255] to [0,1] tensor
        transforms.Normalize(  # Normalize to [-1, 1] range
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
    ])

    # load dataset from file (subfolder names = class labels)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # get class names from subfolder names
    class_names = dataset.classes

    # return datasets and class names
    return dataset, class_names
