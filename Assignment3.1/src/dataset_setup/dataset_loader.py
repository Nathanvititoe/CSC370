import os
from pathlib import Path
import tensorflow as tf


# function to create a dataset, re
def create_dataset(image_label_list, img_size, batch_size, shuffle=False):
    # split path/label tuple into two separate lists
    paths, labels = zip(*image_label_list)

    # create a tf datasset using the img path's and their labels
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    # function to load and preprocess each image
    def process_image(path, label):
        img = tf.io.read_file(path) # read the img file
        img = tf.image.decode_jpeg(img, channels=3) # convert jpg into a tensor
        img = tf.image.resize(img, img_size) # resize all imgs to match our set size (224x224)

        label = tf.cast(label, tf.int32)  # make sure labels are integers
        return img, label  # return processed images w/ their labels
    
    # run every image through the above function
    ds = ds.map(process_image)

    # shuffle the training data and group into batches 
    if shuffle:
        ds = ds.shuffle(7000) # use greater buffer than total images in ds
    ds = ds.batch(batch_size) 
    return ds

    
# builds a list of (image_path, label) tuples
def build_dataset_list(dataset_dir, class_map):
    dataset = []

    # iterate through dataset and each subfolder
    for folder in sorted(os.listdir(dataset_dir)):
        label = class_map.get(folder) # use class map to assign each subfolder an int label

        # skip folders that arent in the class map
        if label is None:
            print(f"Skipping unknown folder: {folder}")
            continue
        
        folder_path = Path(dataset_dir) / folder
        
        # for each image, record the full path and its label
        for img_file in os.listdir(folder_path):
            img_path = folder_path / img_file
            dataset.append((str(img_path), label))

    print(f"\nLoaded {len(dataset)} images.") # output total image count
    return dataset

# function to get the training and validation sets
def get_datasets(train_list, val_list, img_size, batch_size):
    train_ds = create_dataset(train_list,img_size, batch_size, shuffle=True) # create the training ds (shuffle it)
    val_ds = create_dataset(val_list,img_size, batch_size, shuffle=False) # create the validation ds (dont shuffle)
    return train_ds, val_ds # return train/val ds


