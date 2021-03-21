import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def categorize_features() -> dict:
    """Get feature identifiers per feature set, to map features to feature sets.

    Returns
    -------
    feature_cols_dict
        Dictionary with lists of feature identifiers per feature set.
    """
    feature_cols_dict = {}
    for feature_set in ['char', 'word', 'par', 'rest']:
        feature_cols_dict[feature_set] = pd.read_csv(
            f"./data/data/{feature_set}_col.tsv",
            sep='\t', index_col=0, header=None, squeeze=True,
        ).to_list()
    return feature_cols_dict


X_train = pd.read_parquet("./data/data/processed/X_train.parquet")
y_train = pd.read_parquet("./data/data/processed/y_train.parquet").reset_index(drop=True)

X_test = pd.read_parquet("./data/data/processed/X_test.parquet")
y_test = pd.read_parquet("./data/data/processed/y_test.parquet").reset_index(drop=True)

X_val = pd.read_parquet("./data/data/processed/X_val.parquet")
y_val = pd.read_parquet("./data/data/processed/y_val.parquet").reset_index(drop=True)

encoder = LabelEncoder()
encoder.fit(y_train)

y_train_int = encoder.transform(y_train)
y_train_cat = tf.keras.utils.to_categorical(y_train_int)
y_train_cat = np.asarray(y_train_cat)

y_val_int = encoder.transform(y_val)
y_val_cat = tf.keras.utils.to_categorical(y_val_int)
y_val_cat = np.asarray(y_val_cat)

feature_cols = categorize_features()

i = 0

# print this to show that there are bool values in the input
print(X_train[feature_cols['char']].values)


# try and take the bool values out of input
# did not work. Comment out to run other stuff because these take forever
for column in X_train[feature_cols['char']]:
    if X_train[feature_cols['char']].loc[:, column].dtypes == bool:
        X_train[feature_cols['char']].loc[:, column] = X_train[feature_cols['char']].loc[:, column]*1
        print("This is in")

for column in X_train[feature_cols['word']]:
    if X_train[feature_cols['word']].loc[:, column].dtypes == bool:
        X_train[feature_cols['word']].loc[:, column] = X_train[feature_cols['word']].loc[:, column]*1

for column in X_train[feature_cols['par']]:
    if X_train[feature_cols['par']].loc[:, column].dtypes == bool:
        X_train[feature_cols['par']].loc[:, column] = X_train[feature_cols['par']].loc[:, column]*1

for column in X_train[feature_cols['rest']]:
    if X_train[feature_cols['rest']].loc[:, column].dtypes == bool:
        X_train[feature_cols['rest']].loc[:, column] = X_train[feature_cols['rest']].loc[:, column]*1

input_shape1 = 960
input_shape2 = 201
input_shape3 = 400
input_shape4 = 27

# Multi-Input Layers
input1 = keras.Input(shape=input_shape1)
input2 = keras.Input(shape=input_shape2)
input3 = keras.Input(shape=input_shape3)
input4 = keras.Input(shape=input_shape4)

# Hidden Layers
batchNorm = keras.layers.BatchNormalization()(input1)
batchNorm1 = keras.layers.BatchNormalization()(input2)
batchNorm2 = keras.layers.BatchNormalization()(input3)
dense = keras.layers.Dense(300, activation='relu')(batchNorm)
dense2 = keras.layers.Dense(200, activation='relu')(batchNorm1)
dense4 = keras.layers.Dense(400, activation='relu')(batchNorm2)
dropout = keras.layers.Dropout(0.3)(dense)
dropout1 = keras.layers.Dropout(0.3)(dense2)
dropout2 = keras.layers.Dropout(0.3)(dense4)
dense1 = keras.layers.Dense(300, activation='relu')(dropout)
dense3 = keras.layers.Dense(200, activation='relu')(dropout1)
dense5 = keras.layers.Dense(400, activation='relu')(dropout2)
batchNorm3 = keras.layers.BatchNormalization()(input4)
concatenate = keras.layers.Concatenate()([dense1, dense3, dense5, batchNorm3])
batchNorm4 = keras.layers.BatchNormalization()(concatenate)
dense6 = keras.layers.Dense(500, activation='relu')(batchNorm4)
dropout3 = keras.layers.Dropout(0.3)(dense6)
dense7 = keras.layers.Dense(500, activation='relu')(dropout3)

# Output Layer
output = keras.layers.Dense(78, activation='softmax')(dense7)

model = keras.Model(inputs=[input1, input2, input3, input4], outputs=output, name="Sherlock_Reimplementation_Model")

model.summary()

X_test.head()

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# model.fit is when the error comes out. It takes forever but the error eventually comes.
callbacks = [EarlyStopping(monitor="val_loss", patience=5)]
model.fit(
    [
        X_train[feature_cols['char']].values,
        X_train[feature_cols['word']].values,
        X_train[feature_cols['par']].values,
        X_train[feature_cols['rest']].values,
    ],
    y_train_cat,
    validation_data=(
        [
            X_val[feature_cols['char']].values,
            X_val[feature_cols['word']].values,
            X_val[feature_cols['par']].values,
            X_val[feature_cols['rest']].values,
        ],
        y_val_cat
    ),
    callbacks=callbacks, epochs=100, batch_size=256
)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
