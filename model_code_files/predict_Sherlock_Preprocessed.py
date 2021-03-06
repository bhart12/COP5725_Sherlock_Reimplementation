import sys

#import matplotlib.pyplot as plt
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
predicted_labels = mySherlock_model_helper.predict_sherlock(X_test_preprocessed, 'retrained_sherlock_full')

#Predict on our trained model that got score of 0.885
#predicted_labels = mySherlock_model_helper.predict_sherlock(X_test_preprocessed, 'retrained_sherlock_full_together')

#Show scores on predictions
print(f1_score(y_test_preprocessed, predicted_labels, average='weighted'))
