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


#Set variables
i = 0
inc = 10000

print(len(my_X_train.index))
print(my_X_train.size)

#Get first 10000 values from respective dataframes
y_train_subset = my_y_train.head(n=(i + inc)).tail(n=inc)
y_validate_subset = my_y_val.head(n=(i + inc)).tail(n=inc)

#Get first 10000 values from respective dataframes
X_train = my_X_train.head(n=(i + inc)).tail(n=inc)
X_validation = my_X_val.head(n=(i + inc)).tail(n=inc)

print(X_train)

#Train new model on first 10000 values from respective dataframes
mySherlock_model_helper.train_sherlock(X_train, y_train_subset, X_validation, y_validate_subset, model_name='retrained_sherlock_full_split', new_model_flag=False);

#Increment i by inc (which is increment amount)
i += inc

#Continually train model on the next 10000 values until there are less than 10000 values left
while (i + inc) < len(my_X_train.index):
    print("Train on values " + str(i) + " - " + str(i + inc - 1))
    y_train_subset = my_y_train.head(n=(i + inc)).tail(n=inc)
    y_validate_subset = my_y_val.head(n=(i + inc)).tail(n=inc)

    #Get first 2900 X-values in dataframe
    X_train = my_X_train.head(n=(i + inc)).tail(n=inc)
    X_validation = my_X_val.head(n=(i + inc)).tail(n=inc)

    print(X_train)

    mySherlock_model_helper.train_sherlock(X_train, y_train_subset, X_validation, y_validate_subset, model_name='retrained_sherlock_full_split', new_model_flag=True);

    i += inc

#Get values and train model on remaining data
y_train_subset = my_y_train.tail(n=len(my_X_train.index) - i)
y_validate_subset = my_y_val.tail(n=len(my_X_train.index) - i)

X_train = my_X_train.tail(n=len(my_X_train.index) - i)
X_validation = my_X_val.tail(n=len(my_X_train.index) - i)

mySherlock_model_helper.train_sherlock(X_train, y_train_subset, X_validation, y_validate_subset, model_name='retrained_sherlock_full_split', new_model_flag=True);


print("Model trained and saved.")
