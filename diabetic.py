import numpy as np
import pandas as pd

# Read the dataset from the CSV file
data = pd.read_csv("Diabetes.csv")

# Display information about the dataset
data.info()

# Check for any missing values
data.isnull().sum()

# Separate the features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=25)

# Train a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=8, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the Random Forest classifier
from sklearn.metrics import accuracy_score
acc_randomf = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy Random Forest Classifier:", acc_randomf)

# Train a Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the Logistic Regression classifier
acc_logreg = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy Logistic Regression:", acc_logreg)

# Train a K Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the K Neighbors classifier
acc_knn = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy K Neighbors Classifier:", acc_knn)
