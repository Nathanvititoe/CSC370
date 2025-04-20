# load dataset
from sklearn.datasets import load_iris        # type: ignore

# decision tree classifier         
from sklearn.tree import DecisionTreeClassifier  # type: ignore

# split into train/test sets    
from sklearn.model_selection import train_test_split   # type: ignore

# help check model accuracy
from sklearn.metrics import accuracy_score             # type: ignore

# Allows user input 
import numpy as np # type: ignore

# description for user
def introduction():
    print("This machine learning model is designed to predict the species of Iris flower based on its visible characteristics.")
    print("We are using the Scikit-Learn library and their provided decision tree classifier to achieve this.\n")

# function to load the dataset and output details to user
def load_dataset():
    # load the iris dataset
    iris = load_iris()

    # display list of features
    print("These are the features the model will use to predict the Iris flower species:")
    
    # loop through feature names in dataset
    for i, feature in enumerate(iris.feature_names):
        print(f"{i+1}. {feature}") # print them

    # tell user what model can do
    print("\nThe Iris species we can classify are:")
    
    # print species names
    for i, target in enumerate(iris.target_names):
        print(f"{i+1}. {target}")

    return iris # return the dataset


# function to split the dataset
def split_dataset(iris_ds):
    # return split dataset, training/test sets (80/20 split)
    return train_test_split(
        iris_ds.data, iris_ds.target, test_size=0.2, random_state=42
    )

# function to create decision tree and train it
def train_model(train_features, train_labels):
    #train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(train_features, train_labels)

    return clf # return the classifier

# test the model using validation ds
def evaluate(classifier, test_features, test_labels):
    # evaluate the model by testing it with the test ds
    test_prediction = classifier.predict(test_features)

    # Calculate and print accuracy
    accuracy = accuracy_score(test_labels, test_prediction)
    print(f"Test Accuracy : {accuracy:.2f} ({accuracy*100:.1f}%)\n")
    return accuracy

# function to take user input and return a prediction
def user_predict(classifier, iris_ds):
    print("Input measurements for each Iris Flower characteristic and I will give you a prediction!")
    # Get input from user for each feature
    user_input = []
    for feature in iris_ds.feature_names:
        # get user measurements for iris flower
        while True:
            try:
                value = float(input(f"Enter {feature} (in cm): "))
                user_input.append(value)
                break
            except ValueError:
                print(" Invalid input. Please enter a number.")

    # reshape the user input to ensure the model can use it (1 row, 4 features)
    user_input_array = np.array(user_input).reshape(1, -1)

    # make prediction
    prediction = classifier.predict(user_input_array)[0]
    predicted_species = iris_ds.target_names[prediction]

    return predicted_species

