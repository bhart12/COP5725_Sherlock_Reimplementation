import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import tensorflow as tf

from sherlock import helpers
from sherlock.features.preprocessing import extract_features, convert_string_lists_to_lists, prepare_feature_extraction
from sherlock.deploy.train_sherlock import train_sherlock
from sherlock.deploy.predict_sherlock import predict_sherlock

#Get needed data
#helpers.download_data()
prepare_feature_extraction()

train_samples = pd.read_parquet('../data/data/raw/train_values.parquet')
train_labels = pd.read_parquet('../data/data/raw/train_labels.parquet')
validation_samples = pd.read_parquet('../data/data/raw/val_values.parquet')
validation_labels = pd.read_parquet('../data/data/raw/val_labels.parquet')

#Process data (TAKES A WHILE)
train_samples_converted, y_train = convert_string_lists_to_lists(train_samples, train_labels, "values", "type")
validate_samples_converted, y_validate = convert_string_lists_to_lists(validation_samples, validation_labels, "values", "type")

#Get first 2900 y-values in lists
y_train_subset = y_train[:2900]
y_validate_subset = y_validate[:2900]

#Get first 2900 X-values in dataframe and extract features (TAKES A WHILE)
X_train = extract_features(train_samples_converted.head(n=2900))
X_validation = extract_features(validate_samples_converted.head(n=2900))

train_sherlock(X_train, y_train_subset, X_validation, y_train_subset, nn_id='retrained_sherlock_2900Raw');
print("Model trained and saved.")
