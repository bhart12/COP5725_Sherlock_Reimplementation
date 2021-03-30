import sys

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import tensorflow as tf

import mySherlock_model_helper


#model_name = input("Enter model name: ")

#Read parquet data files in at pd dataframe
my_X_train = pd.read_parquet('../data/data/processed/X_train.parquet')
my_y_train = pd.read_parquet('../data/data/processed/y_train.parquet')
my_X_val = pd.read_parquet('../data/data/processed/X_val.parquet')
my_y_val = pd.read_parquet('../data/data/processed/y_val.parquet')

#Train file on all data at once
mySherlock_model_helper.train_sherlock(my_X_train, my_y_train, my_X_val, my_y_val, model_name='retrained_sherlock_full', new_model_flag=False);

print("Model trained and saved.")
