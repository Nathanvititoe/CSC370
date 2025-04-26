import os
from pathlib import Path
import tensorflow as tf

def create_tf_dataset(image_label_list, img_size, batch_size, shuffle=False):
    paths, labels = zip(*image_label_list)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    def process_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        label = tf.cast(label, tf.int32)  # Critical fix
        return img, label

    ds = ds.map(process_image)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    return ds

    
# builds a list of (image_path, label) tuples
def build_dataset_list(dataset_dir, class_map):
    dataset = []

    for folder in sorted(os.listdir(dataset_dir)):
        label = class_map.get(folder)
        if label is None:
            print(f"Skipping unknown folder: {folder}")
            continue

        folder_path = Path(dataset_dir) / folder
        for img_file in os.listdir(folder_path):
            img_path = folder_path / img_file
            dataset.append((str(img_path), label))

    print(f"Loaded {len(dataset)} images.")
    return dataset

def get_loaders(train_list, val_list, img_size, batch_size):
    train_ds = create_tf_dataset(train_list,img_size, batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_list,img_size, batch_size, shuffle=False)
    return train_ds, val_ds


