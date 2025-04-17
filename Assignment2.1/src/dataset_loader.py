import tensorflow as tf

# function to resize images and maintain consistent aspect ratios (padding to fill space)
# def resize_images(images, labels, target_size):
#     # resizes image batch, filling empty space with padding
#     images = tf.map_fn(
#         lambda img: tf.image.resize_with_pad(img, target_size[0], target_size[1]),
#         images
#     )
#     return images, labels

# function to load the dataset from the zip file (dataset directory path, img width/height, number of imgs per batch)
def load_dataset(data_dir, img_size, batch_size):
    # load the training dataset, assign labels from subfolder names
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, # directory path for dataset
        validation_split=0.35, # reserve 20% of data for validation
        subset="training", # loads the training subset
        seed=123, # creates an identical split for consistency
        image_size=img_size, # resizes all images to the passed img_size
        batch_size=batch_size # groups images into batches
    )

    # load the validation dataset, assigning labels from subfolder names
    # uses the same logic as above, but loads the validation subset (20% of data)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.35, 
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names # get class names before prefetch

    # # ensure all images have the same aspect-ratio using padding to fill
    # #    parallel calls allows us to run map operations simultaneously
    # train_ds = train_ds.map(lambda x, y: resize_images(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE) # resize each batch of images while preserving labels
    # val_ds = val_ds.map(lambda x, y: resize_images(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE) # resize each batch of images while preserving labels

    # test adding caching, prefetching for speed
    # if cache:
    #     train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    #     val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    # else:
    #     train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    #     val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


    # return the train and validation datasets, as well as class names (plane types/subfolders)
    return train_ds, val_ds, class_names
