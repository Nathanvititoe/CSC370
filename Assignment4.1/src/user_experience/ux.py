import matplotlib.pyplot as plt
from PIL import Image

# output model description to user
def introduction():
    print("\n--- Intel Image Classifier ---")
    print("This model classifies images into 1/6 categories [buildings, forest, glacier, mountain, sea, street] from the Intel Image Classification dataset.")
    print("Intel Image Classification Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
    print("The model is built using a Convolutional Neural Network (CNN), which automatically learns important features from the training images.")
    print("By recognizing these features, the model can predict the correct type of Image from a set of unseen test images.")

# visualize loss v acc during training and validation 
def visualize_stats(model_history):
    # get metrics from model_history
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5)) # set plot size

    # Plot the training vs. validation Accuracy
    plt.subplot(1,2,1) 
    plt.plot(epochs, acc, label='Training Accuracy') # Training acc plot
    plt.plot(epochs, val_acc, label='Validation Accuracy') # Validation acc plot
    plt.title('Training vs Validation Accuracy') # Title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Accuracy') # y label
    plt.legend() # legend

    # plot training v. validation Loss
    plt.subplot(1,2,2) 
    plt.plot(epochs, loss, label='Training Loss') # training loss plot
    plt.plot(epochs, val_loss, label='Validation Loss') # validation loss plot
    plt.title('Training vs Validation Loss') # title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Loss') # y label
    plt.legend() # legend

    plt.tight_layout() # adjust spacing between the plots
    plt.show() # display them

def visualize_predictions(all_preds, pred_list, class_names):
     # Set up subplots to visualize
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()

    # loop through each image and it's prediction
    for index, (img_path, pred) in enumerate(zip(pred_list, all_preds)):
        class_index = pred.argmax()  # get the predicted class index
        img = Image.open(img_path)  # load image to display

        # plot the image 
        ax = axes[index]
        ax.imshow(img)

        # plot the prediction
        ax.set_title(f"Pred: {class_names[class_index]}")  # set prediction as title
        ax.axis('off')  # hide axis

    plt.tight_layout()  # adjust spacing
    plt.show()