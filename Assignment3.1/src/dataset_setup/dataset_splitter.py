from sklearn.model_selection import train_test_split # type: ignore

def dataset_split(label_list):
   return train_test_split(
        label_list,
        test_size=0.2,
        stratify=[label for _, label in label_list],
        random_state=42
    )
