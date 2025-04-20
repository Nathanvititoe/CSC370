# for image loading & preprocessing
from torchvision import datasets, transforms  # type: ignore 

# for splitting the dataset
from torch.utils.data import random_split, DataLoader      # type: ignore

# function to load and prepare the dataset for the model
def load_dataset(data_dir, img_size, batch_size=16):
    val_split = 0.2
    # preprocessing the images
    transform = transforms.Compose([
        transforms.Resize(img_size), # resize images to match img_size resolution
        transforms.ToTensor(), # convert from [255,255] to [0,1] tensor
        transforms.Normalize(  # Normalize to [-1, 1] range
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
    ])

    # load dataset from file (subfolder names = class labels)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)


    # TEST LOGS
    if len(dataset) == 0:
        raise RuntimeError(f"🚨 No data loaded. Is '{data_dir}' a valid path with labeled subfolders?")
    
    print(f"Loaded {len(dataset)} images from: {data_dir}")
    print(f"Class names: {dataset.classes}")

    # get number of samples for validation/training
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # TEST LOGS
    # Safety checks
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid split: train={train_size}, val={val_size}")

    if batch_size > train_size or batch_size > val_size:
        raise ValueError(f"Batch size {batch_size} too large for split sizes")
    
    # split train/validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # get class names from subfolder names
    class_names = dataset.classes

    # use pytorch dataloaders to load ds using cpu/gpu in parallel and prefetch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4
        )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4 
        )

    # return datasets and class names
    return train_loader, val_loader, class_names



# dont use tf anymore, too complicated to setup for gpu acceleration
# import tensorflow as tf # type: ignore
# from tensorflow.keras.utils import image_dataset_from_directory # type: ignore


# # function to load the dataset from the zip file (dataset directory path, img width/height, number of imgs per batch)
# def load_dataset(data_dir, img_size, batch_size):
#     # load the training dataset, assign labels from subfolder names
#     train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         data_dir, # directory path for dataset
#         validation_split=0.2, # reserve 20% of data for validation
#         subset="training", # loads the training subset
#         seed=123, # creates an identical split for consistency
#         image_size=img_size, # resizes all images to the passed img_size
#         batch_size=batch_size, # groups images into batches
#         label_mode='categorical'

#     )

#     # load the validation dataset, assigning labels from subfolder names
#     # uses the same logic as above, but loads the validation subset (20% of data)
#     val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2, 
#         # subset="validation",
#         seed=123,
#         image_size=img_size,
#         batch_size=batch_size,
#         label_mode='categorical'
#     )
#     class_names = train_ds.class_names # get class names

#     # Apply normalization (rescale to [0, 1])
#     normalization_layer = tf.keras.layers.Rescaling(1./255)
#     train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#     val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#     # Prefetch for performance
#     # loads next batch while curr batch is running, to improve efficiency
#     AUTOTUNE = tf.data.AUTOTUNE
#     train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
#     val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


#     # return the train and validation datasets, as well as class names (plane types/subfolders)
#     return train_ds, val_ds, class_names
