import matplotlib.pyplot as plt
import numpy as np

def introduction():
    print("\n--- Intel Image Classifier ---")
    print("This model classifies images into 1/6 categories [buildings, forest, glacier, mountain, sea, street] from the Intel Image Classification dataset.")
    print("Intel Image Classification Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
    print("The model is built using a Convolutional Neural Network (CNN), which automatically learns important features from the training images.")
    print("By recognizing these features, the model can predict the correct type of Image from a set of unseen test images.")

def visualize_stats():
    print("visualizing")

def visualize_predictions(model, pred_loader, pred_list, class_names):
    # Generate predictions for seg_pred
    all_preds = model.predict(pred_loader)

    # Set up subplots to visualize
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))  # 5x5 grid, adjust as needed
    axes = axes.flatten()

    for index, (img_path, pred) in enumerate(zip(pred_list, all_preds)):
        class_index = pred.argmax()
        img = Image.open(img_path[0])  # Load the image to display
        ax = axes[index]
        ax.imshow(img)
        ax.set_title(f"Pred: {class_names[class_index]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()