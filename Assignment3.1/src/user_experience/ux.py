import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import random

# describe the program
def introduction():
    print("\n--- Aircraft Type Identifier ---")
    print("Given an assortment of aircraft the goal of this machine learning model is to determine whether the aircraft is a Helicopter, Fighter Jet, or Bomber aircraft.")

# visualize loss v acc during training and validation 
def visualize_history(model_history):
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# visualize predictions using the images 
def visualize_predictions(model, val_loader, class_names, num_images=12):
    # Get a batch of images and labels
    images, labels = next(iter(val_loader))
    predictions = model.predict(images)
    print("Predictions shape:", predictions.shape)
    print("Sample predictions:", predictions[0])
    print("Predicted class:", np.argmax(predictions[0]))

    plt.figure(figsize=(15, 10))
    for i in range(min(num_images, len(images))):
        img = images[i]
        img = np.clip(img, 0, 1)  # just to be safe

        true_label = class_names[int(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        color = "green" if true_label == pred_label else "red"

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()