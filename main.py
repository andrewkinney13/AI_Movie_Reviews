# Andrew Kinney
# AI, Project 2
# Using SVM to predict movie score based on review content (words)
# 11.24.2023

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import os

# Returns a list of all the file names in a directory
def get_file_names(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):  # Check if it's a file
            file_names.append(file)
    return file_names

# Returns a list of
def process_review_files(files):

    # Iterate through each file, obtaining the review text
    reviews = []
    for file in files:

        # Obtain the review based on contents of file
        data = open(file, "r")
        try:
            reviews.append(data.read())
        except UnicodeDecodeError:
            pass

    return reviews

# Returns two lists, text for review and label for that review
def get_review_and_label(negative_files, positive_files):
    reviews = []
    labels = []

    # Negative reviews
    reviews += process_review_files(negative_files)
    negative_reviews = len(reviews)
    [labels.append(0) for _ in range(negative_reviews)]

    # Obtain the positive review data
    reviews += process_review_files(positive_files)
    positive_reviews = len(reviews) - negative_reviews
    [labels.append(1) for _ in range(positive_reviews)]

    # Return the reviews and labels
    return (reviews, labels)

# Obtain the files 
trainingNegativeFiles = ["aclImdb/train/neg" + "/" + file for file in get_file_names("aclImdb/train/neg")]
trainingPositiveFiles = ["aclImdb/train/pos" + "/" + file for file in get_file_names("aclImdb/train/pos")]
testingNegativeFiles = ["aclImdb/test/neg" + "/" + file for file in get_file_names("aclImdb/test/neg")]
testingPositiveFiles = ["aclImdb/test/pos" + "/" + file for file in get_file_names("aclImdb/test/pos")]

# Assign review label lists w/ file data
trainingReviews, trainingLabels = get_review_and_label(trainingNegativeFiles, trainingPositiveFiles)
testingReviews, testingLables = get_review_and_label(testingNegativeFiles, testingPositiveFiles)

# Convert text data into numerical vectors to get Xs
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(trainingReviews)  
X_test = vectorizer.transform(testingReviews)  

# Obtain Ys from the labsl
y_train, y_test = trainingLabels, testingLables

"""
# Sample dataset
reviews = [
"Love this product, would buy again!",
"Not what I expected, quite disappointing.",
"Amazing quality, fast delivery.",
"The product broke after one use.",
"Great value for money."
]
labels = [1, 0, 1, 0, 1] # 1 for positive, 0 for negative

# Convert text data into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)   # x are the words in the review, numerically

# Since the dataset is very small, we use it all for training (not recommended for larger datasets)
X_train, X_test, y_train, y_test = X, X, labels, labels # y are the actual review results, aka good or neg
"""

# Create an SVM classifier
classifier = svm.SVC(kernel='linear', verbose= True)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
accuracy = round(accuracy, 2)

# Output to text file
output = "Accuracy: " + str(accuracy)
outputFile = open("out.txt", "w")
outputFile.write(output)
outputFile.close()
