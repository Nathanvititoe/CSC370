import os
from pathlib import Path
import librosa
import numpy as np
import tensorflow as tf
from collections import defaultdict

# CREATE A VISUALIZATION OF THE IMAGE WE CREATE FROM THE WAV FILES

# build a list of (file, label) from the provided CSV
def build_dataset_list(DATASET_DIR, metadata_df, folds, sample_rate):
    # init variables
    dataset = []  # list of (path, label) pairs
    file_durations = []  # list duration of each audio file
    file_sizes = []      # list size of each audio file
    files_per_fold = defaultdict(int)  # number of files per fold

    print(f"\n Loading UrbanSound files from folds {folds}...")

    # iterate through each folder/subfolder/file
    for _, row in metadata_df.iterrows():
        if row["fold"] in folds:  
            file_path = Path(DATASET_DIR) / f"fold{row['fold']}" / row["slice_file_name"] # get full path
            label = int(row["classID"]) # get label for the file
            fold = int(row["fold"])      # get fold number
            
            # skip files that are in the csv but not in the dataset
            if not file_path.exists():
                print(f" Skipping Unknown file: {file_path}")
                continue
            # load audio file and its stats
            try:
                y, _ = librosa.load(file_path, sr=sample_rate)  # load audio
                duration = librosa.get_duration(y=y, sr=sample_rate)  # get file duration(seconds)
                size_bytes = os.path.getsize(file_path)      # get file size in bytes
    
                file_durations.append(duration)     # add duration to list
                file_sizes.append(size_bytes)       # add size to list
                files_per_fold[row["fold"]] += 1    # increment count of files in this fold

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue  # skip files that throw errors

            dataset.append((str(file_path), label, fold)) # add file to dataset

    # GRAPH THE FILE SIZES/DURATIONS TO LOOK FOR OUTLIERS
    total_files = len(dataset) # get length of dataset
    avg_duration = np.mean(file_durations) if file_durations else 0 # calc the avg file duration
    avg_size_kb = np.mean(file_sizes) / 1024 if file_sizes else 0 # calc the avg file size

    # Output stats
    print(f"\n Loaded {total_files} files")
    print(f"Average duration per file: {avg_duration:.2f} sec")
    print(f"Average file size: {avg_size_kb:.2f} KB")

    print("Files per fold:")
    for fold, count in sorted(files_per_fold.items()):
        print(f"  Fold {fold}: {count} files")
    labels = [label for _, label in dataset] # get list of labels

    return dataset, file_durations, file_sizes, files_per_fold, labels

# convert each audio file(.wav) into a mel spectrogram image
def preprocess_audio(file_path, label, sample_rate, duration, n_mels=64):
    file_path = file_path.numpy().decode("utf-8") # convert filepath to utf8

    y, _ = librosa.load(file_path, sr=sample_rate, duration=duration) # load audio using librosa  
    
    # trim audio to fixed length
    if len(y) < sample_rate * duration:
        y = np.pad(y, (0, sample_rate * duration - len(y)))
    else:
        y = y[:sample_rate * duration]

    # convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y, sample_rate=sample_rate, n_mels=n_mels) # get the spectrogram values
    mel_spec_in_db = librosa.power_to_db(mel_spec, ref=np.max) # convert to db scale
    
    # ensure type and shape are consistent
    mel_spec_in_db = mel_spec_in_db.astype(np.float32) 
    mel_spec_in_db = np.expand_dims(mel_spec_in_db, axis=-1)  # [time, freq, 1]
    
    return mel_spec_in_db, label

# allow us to preprocess by batch instead of all at once
def build_tf_preprocessor(sample_rate, duration, n_mels):
# wrap the preprocessing with a tensorFlow function so it can run preprocess_audio
    def tf_preprocess_audio(file_path, label):
        # convert audio to spectrogram and its label
        specgram, label = tf.py_function(
            func=preprocess_audio,
            inp=[file_path, label, sample_rate, duration, n_mels],
            Tout=[tf.float32, tf.int32]
        )
        specgram.set_shape([None, None, 1])
        return specgram, label
    return tf_preprocess_audio


# create dataset object from list of paths/labels
def create_dataset(audio_label_list, batch_size, shuffle, sample_rate, duration, n_mels):
    # split into separate lists
    paths, labels = zip(*[(path, label) for path, label, _ in audio_label_list])


    # convert to tensorFlow dataset from lists
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    # convert each file to spectrogram
    tf_preprocess_fn = build_tf_preprocessor(sample_rate, duration, n_mels)
    ds = ds.map(tf_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # apply shuffling when asked
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    # use batching and prefetching for speed
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# get train/val dataset objects
def get_datasets(train_list, val_list, batch_size=32, sample_rate=22050, duration=4, n_mels=64):
    # get training ds
    train_ds = create_dataset(
        train_list, batch_size=batch_size, shuffle=True, sample_rate=sample_rate, duration=duration, n_mels=n_mels
    )

    # get validation ds
    val_ds = create_dataset(
        val_list, batch_size=batch_size, shuffle=False, sample_rate=sample_rate, duration=duration, n_mels=n_mels
    )
    return train_ds, val_ds
