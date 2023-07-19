# Diabetes Detection

This repository contains code for training and evaluating classifiers to detect diabetes using the Pima Indian Diabetes dataset.

## Dataset

The dataset used for training and testing the models is stored in the `Diabetes.csv` file. It contains various features such as glucose level, blood pressure, body mass index, etc., and a target variable indicating the presence or absence of diabetes.

## Data Preprocessing

The following preprocessing steps are performed on the dataset:

- Checking for Missing Values: The code checks for any missing values in the dataset using `isnull().sum()`.
- Splitting into Features and Target: The features (X) and target variable (y) are separated using `iloc`.

## Model Training and Evaluation

The code trains and evaluates three classifiers on the dataset: Random Forest, Logistic Regression, and K Neighbors.

- Random Forest Classifier: The code uses `RandomForestClassifier` from the `sklearn.ensemble` module to train a Random Forest classifier with 8 estimators and entropy criterion. The accuracy of the classifier is calculated using `accuracy_score`.
- Logistic Regression Classifier: The code uses `LogisticRegression` from the `sklearn.linear_model` module to train a Logistic Regression classifier with lbfgs solver and 1000 maximum iterations. The accuracy of the classifier is calculated using `accuracy_score`.
- K Neighbors Classifier: The code uses `KNeighborsClassifier` from the `sklearn.neighbors` module to train a K Neighbors classifier with 8 neighbors. The accuracy of the classifier is calculated using `accuracy_score`.

## Usage

To run the diabetes detection code, follow these steps:

1. Make sure you have the necessary dependencies installed. You can install them using `pip`:
2. Ensure that the `Diabetes.csv` dataset file is located in the same directory as the code file.
3. Run the code file, and the accuracy of each classifier will be printed to the console.

## Dependencies

The following dependencies are required to run the code:

- numpy
- pandas
- scikit-learn

You can install the dependencies by running the following command:
pip install numpy pandas scikit-learn

