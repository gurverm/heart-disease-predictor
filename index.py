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