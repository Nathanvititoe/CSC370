import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def create_dataframe_dataset(data_path):
    images, labels = [], []
    classes = os.listdir(data_path)
    for label in classes:
        class_dir = os.path.join(data_path, label)
        for file in os.listdir(class_dir):
            images.append(os.path.join(class_dir, file))
            labels.append(label)
    df = pd.DataFrame({"file_paths": images, "labels": labels})
    return df

def get_generators(data_path, img_size, batch_size):
    df = create_dataframe_dataset(data_path)
    train_df, dummy = train_test_split(df, test_size=0.2, random_state=44)
    val_df, test_df = train_test_split(dummy, test_size=0.5, random_state=44)

    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    test_gen = ImageDataGenerator(rescale=1.0/255)

    train_ds = train_gen.flow_from_dataframe(
        train_df,
        x_col="file_paths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = test_gen.flow_from_dataframe(
        val_df,
        x_col="file_paths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )
    test_ds = test_gen.flow_from_dataframe(
        test_df,
        x_col="file_paths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )
    return train_ds, val_ds, list(train_ds.class_indices.keys())