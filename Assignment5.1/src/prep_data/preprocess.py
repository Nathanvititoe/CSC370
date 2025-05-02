# suppress warnings
import warnings
import os
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# external libraries
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import scipy.signal
import soundfile as sf
from tqdm import tqdm

# load YamNet as pretrained model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
yamnet_model.trainable = False  # freeze the model (prevent training)


def load_file(filepath, sample_rate, audio_length):
    audio, sr = sf.read(filepath) # load audio files using soundfile
    
    # ensure all files are monophonic
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # stereo to mono

    # ensure all files use the same sample rate
    if sr != sample_rate:
        num_samples = int(sample_rate * len(audio) / sr) # technical audio length (samples per sec / # of seconds)
        audio = scipy.signal.resample(audio, num_samples)

    # Pad or trim
    if len(audio) < audio_length:
        pad = audio_length - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    else:
        audio = audio[:audio_length]

    return tf.convert_to_tensor(audio, dtype=tf.float32)

# get feature vector to use for 
def extract_embedding(waveform):
    #TODO: create spectrogram and waveform visualization: HERE

    # use yamnet to get 1024-embeddings and spectrograms
    scores, embeddings, spectrogram = yamnet_model(waveform) 
    return tf.reduce_mean(embeddings, axis=0) # get mean to create feature vector 

# get all labels and embeddings for files
def load_data_from_folds(audio_path, csv_path, sample_rate, target_num_samples):
    df = pd.read_csv(csv_path) # read the csv to get dataframe
    filepaths, labels = [], [] # init lists

    # define tqdm progress bar for ui
    progress_bar = tqdm( df.iterrows(),
        total=len(df), # dataframe length
        ncols=100, # progress bar width
        desc="Loading Files...", # prefix
        bar_format='{desc:10}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}' # format progress bar
    )

    # iterate through dataframe via tqdm
    for _, row in progress_bar: 
        path = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name']) # get full file path
        try: 
            waveform = load_file(path, sample_rate, target_num_samples) # load the .wav file
            embedding = extract_embedding(waveform) # get the embedding by passing file through yamnet
            filepaths.append(embedding.numpy()) # add the embedding mean to the list
            labels.append(row['classID']) # add the label to the list 
        
        # throw error if preprocessing fails
        except Exception as e: 
            print(f"Skipping {path}: {e}")

    return np.stack(filepaths), np.array(labels) # return features/labels