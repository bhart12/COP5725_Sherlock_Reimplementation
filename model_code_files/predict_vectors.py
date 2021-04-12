import sys

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import tensorflow as tf

import mySherlock_model_helper, train_column_vectors


#Read parquet data files in at pd dataframe
my_X_train = pd.read_parquet('./data/data/processed/X_train.parquet')
my_y_train = pd.read_parquet('./data/data/processed/y_train.parquet')
my_X_val = pd.read_parquet('./data/data/processed/X_val.parquet')
my_y_val = pd.read_parquet('./data/data/processed/y_val.parquet')

X_test_preprocessed = pd.read_parquet("./data/data/processed/X_test.parquet")
y_test_preprocessed = pd.read_parquet("./data/data/processed/y_test.parquet").reset_index(drop=True)

#train each column vector model
#train_column_vectors.train_sherlock_char(my_X_train, my_y_train, my_X_val, my_y_val, model_name='model_char', new_model_flag=True)
#train_column_vectors.train_sherlock_word(my_X_train, my_y_train, my_X_val, my_y_val, model_name='model_word', new_model_flag=True)
#train_column_vectors.train_sherlock_par(my_X_train, my_y_train, my_X_val, my_y_val, model_name='model_par', new_model_flag=True)
#train_column_vectors.train_sherlock_rest(my_X_train, my_y_train, my_X_val, my_y_val, model_name='model_rest', new_model_flag=True)

#Predict on preprocessed data
predicted_char = train_column_vectors.predict_sherlock_col(X_test_preprocessed, 'model_char', 'char')
predicted_word = train_column_vectors.predict_sherlock_col(X_test_preprocessed, 'model_word', 'word')
predicted_par = train_column_vectors.predict_sherlock_col(X_test_preprocessed, 'model_par', 'par')
predicted_rest = train_column_vectors.predict_sherlock_col(X_test_preprocessed, 'model_rest', 'rest')

#Show scores on predictions
print(f1_score(y_test_preprocessed, predicted_char, average='weighted'))
print(f1_score(y_test_preprocessed, predicted_word, average='weighted'))
print(f1_score(y_test_preprocessed, predicted_par, average='weighted'))
print(f1_score(y_test_preprocessed, predicted_rest, average='weighted'))