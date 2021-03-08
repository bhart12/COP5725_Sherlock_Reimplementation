import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import tensorflow as tf

import mySherlock_model_helper

from sherlock.features.preprocessing import prepare_feature_extraction
'''
from sherlock import helpers
from sherlock.features.preprocessing import extract_features, convert_string_lists_to_lists, prepare_feature_extraction
from sherlock.deploy.train_sherlock import train_sherlock
from sherlock.deploy.predict_sherlock import predict_sherlock
'''

#model_name = input("Enter model name: ")

#Get needed data
#helpers.download_data()
prepare_feature_extraction()

#Read parquet data files in at pd dataframe
my_X_train = pd.read_parquet('../data/data/processed/X_train.parquet')
my_y_train = pd.read_parquet('../data/data/processed/y_train.parquet')
my_X_val = pd.read_parquet('../data/data/processed/X_val.parquet')
my_y_val = pd.read_parquet('../data/data/processed/y_val.parquet')

#Get first 2900 y-values in lists
y_train_subset = my_y_train.head(n=2900)
y_validate_subset = my_y_val.head(n=2900)

#Get first 2900 X-values in dataframe
X_train = my_X_train.head(n=2900)
X_validation = my_X_val.head(n=2900)

mySherlock_model_helper.train_sherlock(X_train, y_train_subset, X_validation, y_train_subset, model_name='retrained_sherlock_2900_preproc');

#train_sherlock(X_train, y_train_subset, X_validation, y_train_subset, nn_id=model_name);

#train_sherlock(X_train.head(n=1450), y_train_subset.head(n=1450), X_validation.head(n=1450), y_train_subset.head(n=1450), nn_id='retrained_sherlock_2900halves_preproc_');
#train_sherlock(X_train.tail(n=1450), y_train_subset.tail(n=1450), X_validation.tail(n=1450), y_train_subset.tail(n=1450), nn_id='retrained_sherlock_2900halves_preproc_');
print("Model trained and saved.")
