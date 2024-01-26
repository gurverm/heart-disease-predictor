# import libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data collection and processessing
# load the csv data to a Pandas DataFrame

heart_data = pd.read_csv('./data/heart_disease_data.csv')

# print first 5 rows of the data set
print(heart_data.head())

# print last 5 rows
print(heart_data.tail())

# number of rows and columns in the dataset
print(heart_data.shape)

# getting more information about the data
print(heart_data.info())

# checking for missing values
print(heart_data.isnull().sum())

# statistical measures of the data
print(heart_data.describe())

# checking the distribution of target variables
# 1: represents defective heart, 0: represents healthy heart
print(heart_data['target'].value_counts()) 

# splitting the features and target
X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

print(X)
print(Y)


# splitting data into training and test data
# stratify distributes the target values evenly
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# model training using logistic regression
model = LogisticRegression(max_iter=1000)
# training logistic regression model with training data
# fit will find the relationship between these features and the corresponding targets
model.fit(X_train, Y_train)

# model evaluation using accuracy score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data: ', training_data_accuracy) #outcome of ~85%

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data: ', accuracy_score(X_test_prediction, Y_test)) #outcome of ~80%

