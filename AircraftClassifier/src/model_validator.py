import torch # type: ignore
from torch import nn # type: ignore
from skorch import NeuralNetClassifier # type: ignore
from src.dataset_loader import load_dataset
from src.model_builder import build_model
from sklearn.model_selection import train_test_split # type: ignore

# function to validate the trained model's accuracy using the validation dataset
def model_validator(data_dir, img_size, batch_size):
    # Load validation dataset only
    dataset, class_names = load_dataset(data_dir, img_size)
    y_labels = torch.tensor(dataset.targets)

    # Load model architecture
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load("trainedModels/aircraft_classifier.pt"))
    model.eval()

    # Evaluate on the validation half (manually split to match training logic)
    indices = list(range(len(dataset)))
    _, val_idx = train_test_split(indices, test_size=0.5, stratify=y_labels, random_state=42)
    X_val = torch.utils.data.Subset(dataset, val_idx)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    loss_total = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_total += loss.item() * labels.size(0)

    accuracy = correct / total
    avg_loss = loss_total / total

    print(f"\nValidation Accuracy: {accuracy:.2%}")
    print(f"\nValidation Loss (confidence): {avg_loss:.2f}\n Loss > 3.5 is considered guessing")
