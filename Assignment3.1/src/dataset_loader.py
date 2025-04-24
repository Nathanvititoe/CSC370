import os
from pathlib import Path
from torch.utils.data import Dataset # type: ignore
from PIL import Image # type: ignore
import torchvision.transforms as T # type: ignore
from torch.utils.data import DataLoader # type: ignore

class AircraftImageDataset(Dataset):
    def __init__(self, image_label_list, img_size=(96, 96)):
        self.data = image_label_list
        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),  # returns shape [C,H,W]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label
    
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

def get_loaders(train_list, val_list):
    train_dataset = AircraftImageDataset(train_list)
    val_dataset = AircraftImageDataset(val_list)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader


