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

#Read parquet data files in at pd dataframe
test_samples = pd.read_parquet('../data/data/raw/test_values.parquet')
test_labels = pd.read_parquet('../data/data/raw/test_labels.parquet')

#Process first 100 entries of data
test_samples_converted, y_test = convert_string_lists_to_lists(test_samples, test_labels, "values", "type")
y_test_subset = y_test[:100]
X_test = extract_features(test_samples_converted.head(n=100))

#Predict
predicted_labels = predict_sherlock(X_test, nn_id='retrained_sherlock')

#Show scores
print(f1_score(y_test_subset, predicted_labels, average="weighted"))
