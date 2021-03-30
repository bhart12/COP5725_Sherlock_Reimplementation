import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import tensorflow as tf

import mySherlock_model_helper


#Read parquet data files in at pd dataframe
X_test_preprocessed = pd.read_parquet("../data/data/processed/X_test.parquet")
y_test_preprocessed = pd.read_parquet("../data/data/processed/y_test.parquet").reset_index(drop=True)

#Predict on preprocessed data
predicted_labels = mySherlock_model_helper.predict_sherlock(X_test_preprocessed, 'retrained_sherlock_full_split')

#Show scores on predictions
print(f1_score(y_test_preprocessed, predicted_labels, average='weighted'))

#Predict on first 10000 elements of data if your machines RAM is insufficient
'''
#Predict on first 10000 elements of data
predicted_labels = mySherlock_model_helper.predict_sherlock(X_test_preprocessed.head(n=10000), 'retrained_sherlock_full_split')

#Show scores on first 10000 elements of data
print(f1_score(y_test_preprocessed.head(n=10000), predicted_labels, average='weighted'))
'''
