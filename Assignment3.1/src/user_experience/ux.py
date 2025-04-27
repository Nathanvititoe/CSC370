import matplotlib.pyplot as plt

# describe the program
def introduction():
    print("\n--- Aircraft Type Identifier ---")
    print("Given an assortment of aircraft the goal of this machine learning model is to determine whether the aircraft is a Helicopter, Fighter Jet, or Bomber aircraft.")
    print("The model is built using a Convolutional Neural Network (CNN), which automatically learns important features (like shape, wings, and rotors) from the training images.")
    print("By recognizing these features, the model can predict the correct type of aircraft from a set of unseen test images.")
    print("The model is trained using a subset of this Kaggle Dataset: \n")
    print("Military Aircraft Detection Dataset: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset \n")
    print("This is the subset dataset I've used: \n")
    print("Military Aircraft Detection Subset: https://www.kaggle.com/datasets/nathanvititoe/military-aircraft-datasetsubset \n")


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