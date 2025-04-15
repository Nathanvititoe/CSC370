import tensorflow as tf

# function to load the dataset from the zip file (dataset directory path, img width/height, number of imgs per batch)
def load_dataset(data_dir, img_size=(180, 180), batch_size=32):
    # load the training dataset, assign labels from subfolder names
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, # directory path for dataset
        validation_split=0.2, # reserve 20% of data for validation
        subset="training", # loads the training subset
        seed=123, # creates an identical split for consistency
        image_size=img_size, # resizes all images to the passed img_size
        batch_size=batch_size # groups images into batches
    )

    # load the validation dataset, assigning labels from subfolder names
    # uses the same logic as above, but loads the validation subset (20% of data)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2, 
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # return the train and validation datasets, as well as class names (plane types/subfolders)
    return train_ds, val_ds, train_ds.class_names
