import os
import tensorflow as tf
import pandas as pd


TARGET_SR = 16000
DURATION_SEC = 4
AUDIO_LEN = TARGET_SR * DURATION_SEC
#TODO: add comments to these functions

def is_valid_wav(filepath):
    try:
        _ = tf.audio.decode_wav(tf.io.read_file(filepath))
        return True
    except tf.errors.InvalidArgumentError:
        return False

def decode_audio(filepath):
    audio_binary = tf.io.read_file(filepath)
    audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # shape: (samples,)
    audio = tf.cast(audio, tf.float32)

    # If audio is longer than target length, trim
    audio = audio[:AUDIO_LEN]

    # If audio is shorter, pad with zeros
    padding = tf.maximum(0, AUDIO_LEN - tf.shape(audio)[0])
    audio = tf.pad(audio, [[0, padding]])

    return tf.expand_dims(audio, axis=-1)  # shape: (samples, 1)

def load_metadata(csv_path, audio_dir, folds_to_include=None):
    metadata = pd.read_csv(csv_path)
    if folds_to_include is not None:
        metadata = metadata[metadata['fold'].isin(folds_to_include)]
    filepaths = metadata.apply(
        lambda row: os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name']),
        axis=1
    )
    labels = metadata['classID']
    return list(filepaths), list(labels)

def create_dataset(filepaths, labels, batch_size=32, shuffle=True):
    files_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_ds = files_ds.map(decode_audio, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((audio_ds, labels_ds))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset