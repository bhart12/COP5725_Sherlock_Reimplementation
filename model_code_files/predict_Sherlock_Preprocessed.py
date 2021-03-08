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
X_test_preprocessed = pd.read_parquet("../data/data/processed/X_test.parquet")
y_test_preprocessed = pd.read_parquet("../data/data/processed/y_test.parquet").reset_index(drop=True)

#Predict on first 25 elements of data
predicted_labels = predict_sherlock(X_test_preprocessed.head(n=25), 'retrained_sherlock_full_preproc')

#Show scores on first 25 elements of data
print(f1_score(y_test_preprocessed.head(n=25), predicted_labels, average='weighted'))
