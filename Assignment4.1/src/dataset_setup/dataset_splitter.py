from sklearn.model_selection import train_test_split 

# split the dataset into training and validation sets
def split_dataset(dataset_list):
    paths, labels = zip(*dataset_list)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=None
    )
    # combine paths and labels
    train_list = list(zip(train_paths, train_labels))
    val_list = list(zip(val_paths, val_labels))
    
    return train_list, val_list