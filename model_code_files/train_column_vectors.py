import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json

import mySherlock_model_helper

def create_sherlock_model_char(model_name: str):
    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    input_shape_word = 960

    # Multi-Input Layers
    input_word = keras.Input(shape=input_shape_word)

    # Hidden Layers
    batchNorm = keras.layers.BatchNormalization()(input_word)
    dense = keras.layers.Dense(300, activation='relu')(batchNorm)
    dropout = keras.layers.Dropout(0.3)(dense)
    dense1 = keras.layers.Dense(300, activation='relu')(dropout)
    dense6 = keras.layers.Dense(500, activation='relu')(dense1)
    dropout3 = keras.layers.Dropout(0.3)(dense6)
    dense7 = keras.layers.Dense(500, activation='relu')(dropout3)

    # Output Layer
    output = keras.layers.Dense(78, activation='softmax')(dense7)

    char_model = keras.Model(inputs=input_word, outputs=output,
                                 name="Sherlock_Reimplementation_Model")

    char_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return char_model, callbacks



def train_sherlock_char(
    X_train_values: pd.DataFrame,
    y_train_values: list,
    X_val_values: pd.DataFrame,
    y_val_values: list,
    model_name: str,
    new_model_flag: bool,
):
    if model_name == "sherlock":
        raise ValueError(
        )

    feature_column_names = mySherlock_model_helper.get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = \
        mySherlock_model_helper._convert_list_of_labels_to_categorical_label_encodings(y_train_values,
                                                                                       y_val_values, model_name)
    sherlock_model, callbacks = create_sherlock_model_char(model_name)

    print("Successfully loaded and compiled model, now fitting model on data.")
    print("My file training!")

    sherlock_model.fit(
        np.asarray(X_train_values[feature_column_names['char']].values).astype(np.float32),
        y_train_categories,
        validation_data=(np.asarray(X_val_values[feature_column_names['char']].values).astype(np.float32),
                         y_val_categories),
        callbacks=callbacks, epochs=100, batch_size=256
    )

    mySherlock_model_helper._save_trained_model(sherlock_model, model_name)

    print('Retrained Sherlock with Word Feature Vectors.')


def create_sherlock_model_word(model_name: str):
    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    input_shape_word = 201

    #Input layer for word column vector
    input_word = keras.Input(shape=input_shape_word)

    # Hidden Layers
    batchNorm1 = keras.layers.BatchNormalization()(input_word)
    dense2 = keras.layers.Dense(200, activation='relu')(batchNorm1)
    dropout1 = keras.layers.Dropout(0.3)(dense2)
    dense3 = keras.layers.Dense(200, activation='relu')(dropout1)
    dense6 = keras.layers.Dense(500, activation='relu')(dense3)
    dropout3 = keras.layers.Dropout(0.3)(dense6)
    dense7 = keras.layers.Dense(500, activation='relu')(dropout3)

    # Output Layer
    output = keras.layers.Dense(78, activation='softmax')(dense7)

    word_model = keras.Model(inputs=input_word, outputs=output, name="Sherlock_Reimplementation_Model")

    word_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return word_model, callbacks



def train_sherlock_word(
    X_train_values: pd.DataFrame,
    y_train_values: list,
    X_val_values: pd.DataFrame,
    y_val_values: list,
    model_name: str,
    new_model_flag: bool,
):
    if model_name == "sherlock":
        raise ValueError(
        )

    feature_column_names = mySherlock_model_helper.get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = \
        mySherlock_model_helper._convert_list_of_labels_to_categorical_label_encodings(y_train_values,
                                                                                       y_val_values, model_name)
    sherlock_model, callbacks = create_sherlock_model_word(model_name)

    print("Successfully loaded and compiled model, now fitting model on data.")
    print("My file training!")

    sherlock_model.fit(
        np.asarray(X_train_values[feature_column_names['word']].values).astype(np.float32),
        y_train_categories,
        validation_data=(np.asarray(X_val_values[feature_column_names['word']].values).astype(np.float32),
                         y_val_categories),
        callbacks=callbacks, epochs=100, batch_size=256
    )

    mySherlock_model_helper._save_trained_model(sherlock_model, model_name)

    print('Retrained Sherlock with Word Feature Vectors.')


def create_sherlock_model_par(model_name: str):
    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    input_shape_par = 400

    # Multi-Input Layers
    input_par = keras.Input(shape=input_shape_par)

    # Hidden Layers
    batchNorm = keras.layers.BatchNormalization()(input_par)
    dense = keras.layers.Dense(300, activation='relu')(batchNorm)
    dropout = keras.layers.Dropout(0.3)(dense)
    dense1 = keras.layers.Dense(300, activation='relu')(dropout)
    dense6 = keras.layers.Dense(500, activation='relu')(dense1)
    dropout3 = keras.layers.Dropout(0.3)(dense6)
    dense7 = keras.layers.Dense(500, activation='relu')(dropout3)

    # Output Layer
    output = keras.layers.Dense(78, activation='softmax')(dense7)

    par_model = keras.Model(inputs=input_par, outputs=output,
                                 name="Sherlock_Reimplementation_Model")

    par_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return par_model, callbacks



def train_sherlock_par(
    X_train_values: pd.DataFrame,
    y_train_values: list,
    X_val_values: pd.DataFrame,
    y_val_values: list,
    model_name: str,
    new_model_flag: bool,
):
    if model_name == "sherlock":
        raise ValueError(
        )

    feature_column_names = mySherlock_model_helper.get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = \
        mySherlock_model_helper._convert_list_of_labels_to_categorical_label_encodings(y_train_values,
                                                                                       y_val_values, model_name)
    sherlock_model, callbacks = create_sherlock_model_par(model_name)

    print("Successfully loaded and compiled model, now fitting model on data.")
    print("My file training!")

    sherlock_model.fit(
        np.asarray(X_train_values[feature_column_names['par']].values).astype(np.float32),
        y_train_categories,
        validation_data=(np.asarray(X_val_values[feature_column_names['par']].values).astype(np.float32),
                         y_val_categories),
        callbacks=callbacks, epochs=100, batch_size=256
    )

    mySherlock_model_helper._save_trained_model(sherlock_model, model_name)

    print('Retrained Sherlock with Word Feature Vectors.')


def create_sherlock_model_rest(model_name: str):
    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    input_shape_rest = 27

    # Multi-Input Layers
    input_rest = keras.Input(shape=input_shape_rest)

    # Hidden Layers
    batchNorm = keras.layers.BatchNormalization()(input_rest)
    dense = keras.layers.Dense(300, activation='relu')(batchNorm)
    dropout = keras.layers.Dropout(0.3)(dense)
    dense1 = keras.layers.Dense(300, activation='relu')(dropout)
    dense6 = keras.layers.Dense(500, activation='relu')(dense1)
    dropout3 = keras.layers.Dropout(0.3)(dense6)
    dense7 = keras.layers.Dense(500, activation='relu')(dropout3)

    # Output Layer
    output = keras.layers.Dense(78, activation='softmax')(dense7)

    rest_model = keras.Model(inputs=input_rest, outputs=output,
                                 name="Sherlock_Reimplementation_Model")

    rest_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return rest_model, callbacks



def train_sherlock_rest(
    X_train_values: pd.DataFrame,
    y_train_values: list,
    X_val_values: pd.DataFrame,
    y_val_values: list,
    model_name: str,
    new_model_flag: bool,
):
    if model_name == "sherlock":
        raise ValueError(
        )

    feature_column_names = mySherlock_model_helper.get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = \
        mySherlock_model_helper._convert_list_of_labels_to_categorical_label_encodings(y_train_values,
                                                                                       y_val_values, model_name)
    sherlock_model, callbacks = create_sherlock_model_rest(model_name)

    print("Successfully loaded and compiled model, now fitting model on data.")
    print("My file training!")

    sherlock_model.fit(
        np.asarray(X_train_values[feature_column_names['rest']].values).astype(np.float32),
        y_train_categories,
        validation_data=(np.asarray(X_val_values[feature_column_names['rest']].values).astype(np.float32),
                         y_val_categories),
        callbacks=callbacks, epochs=100, batch_size=256
    )

    mySherlock_model_helper._save_trained_model(sherlock_model, model_name)

    print('Retrained Sherlock with Word Feature Vectors.')


def predict_sherlock_col(X_pred_values: pd.DataFrame, model_name: str, col: str) -> np.array:
    #Get model based off of model_name
    sherlock_model, callbacks = mySherlock_model_helper.create_sherlock_model(model_name, True)
    #Get feature names by category in dict
    feature_column_names = mySherlock_model_helper.get_processed_feature_category_dicts()

    #Predict labels
    y_predictions = sherlock_model.predict(
        [
            np.asarray(X_pred_values[feature_column_names[col]].values).astype(np.float32),
        ]
    )

    #Use LabelEncoder to convert y_pred to semantic labels
    y_pred_int = np.argmax(y_predictions, axis=1)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        f"./sherlock/deploy/classes_{model_name}.npy",
        allow_pickle=True
    )
    y_predictions = encoder.inverse_transform(y_pred_int)

    return y_predictions