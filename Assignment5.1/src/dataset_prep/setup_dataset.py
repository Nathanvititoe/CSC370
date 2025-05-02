from src.dataset_prep.dataset_loader import create_dataset,load_metadata, is_valid_wav
# from src.dataset_prep.evaluate_dataset import visualize_audio_stats
from src.ui.visualization import show_spectrogram
import pandas as pd

#TODO: add comments here
def setup_dataset(AUDIO_DIR, CSV_PATH, BATCH_SIZE, val_split=0.2):
    metadata = pd.read_csv(CSV_PATH)  # load csv for label_names
    label_names = list(metadata.drop_duplicates("classID").sort_values("classID")["class"])

    
    all_folds = list(range(1, 11))
    val_count = int(len(all_folds) * val_split)
    val_folds = all_folds[:val_count]
    train_folds = all_folds[val_count:]

    train_paths, train_labels = load_metadata(CSV_PATH, AUDIO_DIR, folds_to_include=train_folds)
    val_paths, val_labels = load_metadata(CSV_PATH, AUDIO_DIR, folds_to_include=val_folds)

    train_filtered = [(p, l) for p, l in zip(train_paths, train_labels) if is_valid_wav(p)]
    val_filtered = [(p, l) for p, l in zip(val_paths, val_labels) if is_valid_wav(p)]

    print(f"Total train files: {len(train_paths)}")
    print(f"Valid train files: {len(train_filtered)}")
    print(f"Total val files: {len(val_paths)}")
    print(f"Valid val files: {len(val_filtered)}")

    if len(train_filtered) == 0:
        raise RuntimeError("No valid training WAV files found.")
    if len(val_filtered) == 0:
        raise RuntimeError("No valid validation WAV files found.")

    # Unpack after checking non-emptiness
    train_paths, train_labels = zip(*train_filtered)
    val_paths, val_labels = zip(*val_filtered)

    train_ds = create_dataset(list(train_paths), list(train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(list(val_paths), list(val_labels), batch_size=BATCH_SIZE, shuffle=False)

    return train_ds, val_ds, label_names