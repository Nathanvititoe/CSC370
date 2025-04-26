import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# describe the program
def introduction():
    print("\n--- Aircraft Type Identifier ---")
    print("Given an assortment of aircraft the goal of this machine learning model is to determine whether the aircraft is a Helicopter, Fighter Jet, or Bomber aircraft.")
    print("The model is built using a Convolutional Neural Network (CNN), which automatically learns important features (like shape, wings, and rotors) from the training images.")
    print("By recognizing these features, the model can predict the correct type of aircraft based on unseen test images.")

# visualize loss v acc during training and validation 
def visualize_history(model_history):
    # get metrics from model_history
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5)) # set plot size

    # Plot the training vs. Validation Accuracy
    plt.subplot(1,2,1) 
    plt.plot(epochs, acc, label='Training Accuracy') # Training acc plot
    plt.plot(epochs, val_acc, label='Validation Accuracy') # Validation acc plot
    plt.title('Training vs Validation Accuracy') # Title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Accuracy') # y label
    plt.legend() # legend

    # Plot Training v. Validation Loss
    plt.subplot(1,2,2) 
    plt.plot(epochs, loss, label='Training Loss') # training loss plot
    plt.plot(epochs, val_loss, label='Validation Loss') # validation loss plot
    plt.title('Training vs Validation Loss') # title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Loss') # y label
    plt.legend() # legend

    plt.tight_layout() # adjust spacing between the plots
    plt.show() # display them

# visualize predictions using the images 
def visualize_predictions(model, val_loader, class_names, num_images=12):
    # Get a batch of images and labels
    images, labels = next(iter(val_loader))

    # get model predictions on sample batch
    predictions = model.predict(images)

    # create plot to display images and predictions
    plt.figure(figsize=(15, 10))

    # loop through sample batch
    for i in range(min(num_images, len(images))):
        img = images[i]
        img = np.clip(img, 0, 1)  # clip pixels to [0,1]

        # output the actual and predicted labels
        true_label = class_names[int(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]

        # color correct/incorrect predictions
        color = "green" if true_label == pred_label else "red"

        plt.subplot(3, 4, i + 1)
        plt.imshow(img) # show the images
        plt.title(f"Pred: {pred_label}\nActual: {true_label}", color=color) # prediction
        plt.axis('off') # turn off axis

    plt.tight_layout() # adjust spacing
    plt.show() # display plot