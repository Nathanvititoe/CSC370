'''
Code a new ML application using Python 3.x.
This program should be able to do simple classification using the ML Scikit Learn library and tell if an incoming plane is a fighter or a bomber.
Use a decision Tree as a classifier. 
Train your classifier based on the data you collect.
An explanation of what this program is should be displayed for the user.
The UX target user is a first-time user.
Directions for the user on the UI.
Over comment your code with your own comments in your own words showing you understand the intent and function of almost every line of code. Make sure to use your own comments.
'''

# radar_classifier.py

# IMPORTS
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# This is just for a clean interface
def intro():
    print("=== WWII Aircraft Radar Classifier ===")
    print("This tool uses a Decision Tree to classify if an incoming aircraft is a Fighter or a Bomber.")
    print("It’s based on radar data and some aircraft characteristics.\n")
    print("Instructions: This is a demo. In a real app, radar would feed in real-time data.\n")

# FEATURES EXAMPLE:
# Let’s assume the features are simplified as follows:
# Speed (km/h), Wingspan (m), Weight (kg), Engine Count

# LABELS:
# 0 = Fighter
# 1 = Bomber

# Example synthetic dataset (because Wikipedia doesn’t give structured data directly)
# These values are based loosely on real WWII aircraft characteristics
data = [
    [580, 11, 3200, 1],  # Fighter: P-51 Mustang
    [560, 10, 2900, 1],  # Fighter: Spitfire
    [450, 22, 16000, 4],  # Bomber: B-17 Flying Fortress
    [430, 28, 18000, 4],  # Bomber: Lancaster
    [600, 12, 3300, 1],  # Fighter: Fw 190
    [400, 31, 17000, 4],  # Bomber: Heinkel He 177
]

labels = [0, 0, 1, 1, 0, 1]  # Fighter = 0, Bomber = 1

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.3, random_state=42
)

# Create a Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf = clf.fit(features_train, labels_train)

# Make predictions using the test data
predictions = clf.predict(features_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(labels_test, predictions)

def main():
    intro()
    print("=== Model Training Complete ===")
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    
    # Try predicting a new aircraft
    print("Try predicting a new plane...")
    new_plane = [440, 24, 15000, 4]  # Example input
    result = clf.predict([new_plane])
    print("Prediction: ", "Bomber" if result[0] == 1 else "Fighter")

if __name__ == "__main__":
    main()
