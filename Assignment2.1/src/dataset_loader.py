import tensorflow as tf

# function to load the dataset from the zip file (dataset directory path, img width/height, number of imgs per batch)
def load_dataset(data_dir, img_size, batch_size):
    # load the training dataset, assign labels from subfolder names
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, # directory path for dataset
        validation_split=0.2, # reserve 20% of data for validation
        subset="training", # loads the training subset
        seed=123, # creates an identical split for consistency
        image_size=img_size, # resizes all images to the passed img_size
        batch_size=batch_size, # groups images into batches
        label_mode='categorical'

    )

    # load the validation dataset, assigning labels from subfolder names
    # uses the same logic as above, but loads the validation subset (20% of data)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.5, 
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Apply normalization (rescale to [0, 1])
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    class_names = train_ds.class_names # get class names

    # return the train and validation datasets, as well as class names (plane types/subfolders)
    return train_ds, val_ds, class_names
