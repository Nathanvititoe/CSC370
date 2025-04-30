import matplotlib.pyplot as plt

# output model description to user
def introduction():
    print("\n\n------------------------------------------------------------------------------------ Intel Image Classifier ------------------------------------------------------------------------------------\n")
    print("Intel Image Classification Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
    print("\nThis model classifies images into 1/6 categories [buildings, forest, glacier, mountain, sea, street] from the Intel Image Classification dataset.")
    print("The model is built using a Convolutional Neural Network (CNN), which automatically learns important features from the training images.")
    print("By recognizing these features, the model can predict the correct type of Image from a set of unseen test images.")
    print("\n---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

# visualize loss v acc during training and validation 
def visualize_stats(classifier):
    model_history = classifier.history
    # get metrics from model_history
    val_acc = [epoch['valid_acc'] for epoch in model_history]
    train_loss = [epoch['train_loss'] for epoch in model_history]
    val_loss = [epoch['valid_loss'] for epoch in model_history]

    if 'train_acc' in model_history[0]:
        train_acc = [epoch['train_acc'] for epoch in model_history]
    else:
        train_acc = None

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12,5)) # set plot size

    # Plot the training vs. validation Accuracy
    plt.subplot(1,2,1) 
    if train_acc:
        plt.plot(epochs, train_acc, label='Training Accuracy') # Training acc plot
    plt.plot(epochs, val_acc, label='Validation Accuracy') # Validation acc plot
    plt.title('Training vs Validation Accuracy') # Title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Accuracy') # y label
    plt.legend() # legend

    # plot training v. validation Loss
    plt.subplot(1,2,2) 
    plt.plot(epochs, train_loss, label='Training Loss') # training loss plot
    plt.plot(epochs, val_loss, label='Validation Loss') # validation loss plot
    plt.title('Training vs Validation Loss') # title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Loss') # y label
    plt.legend() # legend

    plt.tight_layout() # adjust spacing between the plots
    plt.show() # display them