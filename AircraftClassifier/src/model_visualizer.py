import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# function to visualize a batch of predictions (trained model, validation dataset, class names/labels, number of images to show in grid)
def visualize_predictions(model, val_ds, class_names, num_images=9):
    plt.figure(figsize=(12, 12)) # figure size
    for images, labels in val_ds.take(1):  # only use one batch from the validation set for visualization
        preds = model.predict(images) # use the predictions from the selected batch
        pred_labels = np.argmax(preds, axis=1) # get the predicted label for each image
        confidences = np.max(preds, axis=1)  # highest softmax value per prediction

        # loop through images in batch
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1) # create a subplot (3x3 grid)
            plt.imshow(images[i].numpy().astype("uint8"), interpolation='bilinear') # display the image

            actual = class_names[labels[i]] # get actual class name
            predicted = class_names[pred_labels[i]] # get predicted class name
            confidence = confidences[i] # get confidence level

            correct = predicted == actual
           # show prediction and actual (green if correct, red if wrong)
            title_color = "green" if correct else "red"
            plt.title(
                f"Prediction: {predicted} Confidence: ({confidence:.0%})\nActual: {actual}", 
                color=title_color,
                fontsize=10
                )
            plt.axis("off") # turn off axis

    plt.tight_layout() # prevent overlap of images/titles
    plt.savefig("Visualizations/predictions_grid.png", dpi=300) # save grid to file
    plt.show() # display grid visualization
