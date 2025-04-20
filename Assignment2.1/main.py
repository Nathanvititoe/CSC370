import iris_classifier as ic

# main function to control print to user and logic flow
def main_classifier():
    # output description of model for user
    ic.introduction() 

    print("Loading the Dataset...")
    # load the dataset
    iris_data = ic.load_dataset()

    print("Splitting the Dataset...")
    # split the dataset into train and test split (features/labels)
    training_features, test_features, training_labels, test_labels = ic.split_dataset(iris_data)

    print("Training the Model...")
    # use sklearn decision tree to train the model
    model = ic.train_model(training_features, training_labels)

    print("Testing Model Accuracy...")
    # Test the model accuracy using the test features/labels
    ic.evaluate(model, test_features, test_labels)

    print("Your turn!")
    # take user input and provide a prediction
    predictedSpecies = ic.user_predict(model, iris_data)
    print(f"Prediction: {predictedSpecies}")

main_classifier()