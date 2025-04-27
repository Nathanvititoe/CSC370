import torch # type: ignore
import torch.nn as nn # type: ignore
from skorch import NeuralNetClassifier # type: ignore
from skorch.dataset import ValidSplit # type: ignore
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler # type: ignore
from torch._dynamo import OptimizedModule # type: ignore
from src.dataset_loader import load_dataset
from src.model_builder import build_model
from src.model_visualizer import visualize_predictions
from src.model_validator import model_validator
from src.custom_prediction import predict_image
import warnings
import logging
import os

# Clean logs
warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
# for 50/50 split
#   increase dropout, and regularization
#   increase augmenting and tune LR
#   dropout = 0.4, weight_decay = 8e-5, max_epochs = 35

def introduction():
    print("\n--- Military Aircraft Image Classifier ---")
    print("This program uses a convolutional neural network to identify military aircraft types.")
    print("It trains the model using labeled images, evaluates its accuracy, and lets you test predictions.")
    print("\n------------------------------------------------------------\n")

def train_model():
    introduction()

    # Dataset paths
    data_path = "./aircraft_dataset/medium_subset"
    img_size = (124, 124)

    # best params for medium subset
    best_medium_params = {
        "dropout": 0.346,
        "lr": 0.000524,
        "dense_units": 339,
        "weight_decay": 4.97e-5,
        "batch_size": 64,
        "max_epochs": 20
    }
    best_params = best_medium_params
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("\u274C CUDA is required but not available!")

    print("\nStep 1: Loading Dataset and Assigning Labels...\n")
    dataset, class_names = load_dataset(data_path, img_size)
    y_labels = torch.tensor(dataset.targets)

    print(f"Learning on {len(class_names)} aircraft types: {class_names}\n")

    print("Step 2: Building Model...\n")
    model = build_model(num_classes=len(class_names),
                        dropout_rate=best_params["dropout"],
                        dense_units=best_params["dense_units"])

    model: OptimizedModule = torch.compile(model, mode="reduce-overhead", disable=True)

    net = NeuralNetClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=best_params["weight_decay"],
        lr=best_params["lr"],
        max_epochs=best_params["max_epochs"],
        batch_size=best_params["batch_size"],
        device=device,
        train_split=ValidSplit(0.5, stratified=True, random_state=42),
        iterator_train__num_workers=8,
        iterator_train__pin_memory=True,
        iterator_valid__num_workers=8,
        iterator_valid__pin_memory=True,
        verbose=1,
        callbacks=[
            LRScheduler(
                policy='ReduceLROnPlateau',
                monitor='valid_acc',
                factor=0.75,
                patience=4,
                threshold=1e-6,
                cooldown=2,
                min_lr=1e-7
            ),
            EarlyStopping(monitor='valid_acc', patience=5, load_best=True),
            EpochScoring(scoring='accuracy', lower_is_better=False, name='train_acc', on_train=True)
        ]
    )

    print("Step 3: Training Model...\n")
    net.fit(dataset, y=y_labels)

    print("\nStep 4: Saving Model...\n")
    os.makedirs("trainedModels", exist_ok=True)
    torch.save(net.module_.state_dict(), "trainedModels/aircraft_classifier.pt")

    print("\nStep 5: Testing Model Accuracy...\n")
    model_validator(data_path, img_size, best_params["batch_size"])

    print("\nStep 6: Visualizing Predictions...\n")
    visualize_predictions(net, dataset, class_names)

    print("\nTraining and Evaluation Complete!\n")
    predict_image(net, class_names, img_size)

if __name__ == "__main__":
    train_model()
