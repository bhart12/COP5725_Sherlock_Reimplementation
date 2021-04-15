import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json


def create_sherlock_model(model_name: str, created: bool):

    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    #Declare model variable
    model = None

    if created:
        #Open previously trained model
        file = open(f"../models/{model_name}_model.json", "r")
        model = model_from_json(file.read())
        file.close()

        model.load_weights(f"../models/{model_name}_weights.h5")
    else:
        #Create new model from scratch
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return model, callbacks


def _convert_list_of_labels_to_categorical_label_encodings(y_train_values, y_val_values, model_name):

    # Get List of labels from disk
    labels = np.load(f"../sherlock/deploy/classes_labels.npy", allow_pickle=True)

    y_train_int = np.zeros(shape=412059, dtype=int)
    y_val_int = np.zeros(shape=137353, dtype=int)

    # Convert train and test labels to value encodings
    n1 = 0
    for i in labels:
        n2 = 0
        for j in y_train_values['label']:
            if j == i:
                y_train_int[n2] = n1
            n2 += 1
        n1 += 1

    n1 = 0
    for i in labels:
        n2 = 0
        for j in y_val_values['label']:
            if j == i:
                y_val_int[n2] = n1
            n2 += 1
        n1 += 1

    # Convert value encodings to one-hot vectors for the Neural Network
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    return y_train_cat, y_val_cat


def get_processed_feature_category_dicts():

    feature_cols_dict = {}
    for feature_set in ['char', 'word', 'par', 'rest']:
        feature_cols_dict[feature_set] = pd.read_csv(
            f"../sherlock/features/feature_column_identifiers/{feature_set}_col.tsv",
            sep='\t', index_col=0, header=None, squeeze=True,
        ).to_list()
    return feature_cols_dict


def train_sherlock(
    X_train_values: pd.DataFrame,
    y_train_values: list,
    X_val_values: pd.DataFrame,
    y_val_values: list,
    model_name: str,
    new_model_flag: bool,
):

    feature_column_names = get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = _convert_list_of_labels_to_categorical_label_encodings(y_train_values, y_val_values, model_name)
    model, callbacks = create_sherlock_model(model_name, new_model_flag)

    model.fit(
        [
            np.asarray(X_train_values[feature_column_names['char']].values).astype(np.float32),
            np.asarray(X_train_values[feature_column_names['word']].values).astype(np.float32),
            np.asarray(X_train_values[feature_column_names['par']].values).astype(np.float32),
            np.asarray(X_train_values[feature_column_names['rest']].values).astype(np.float32)
        ],
        y_train_categories,
        validation_data=(
            [
                np.asarray(X_val_values[feature_column_names['char']].values).astype(np.float32),
                np.asarray(X_val_values[feature_column_names['word']].values).astype(np.float32),
                np.asarray(X_val_values[feature_column_names['par']].values).astype(np.float32),
                np.asarray(X_val_values[feature_column_names['rest']].values).astype(np.float32)
            ],
            y_val_categories
        ),
        callbacks=callbacks, epochs=100, batch_size=256
    )

    model_json = model.to_json()
    with open(f"../models/{model_name}_model.json", "w") as json:
        json.write(model_json)

    model.save_weights(f"../models/{model_name}_weights.h5")

    print('Retrained Sherlock.')


def predict_sherlock(X_pred_values: pd.DataFrame, model_name: str):

    #Get model based off of model_name
    model, callbacks = create_sherlock_model(model_name, True)
    #Get feature names by category in dict
    feature_column_names = get_processed_feature_category_dicts()

    #Predict labels
    y_predictions = model.predict(
        [
            np.asarray(X_pred_values[feature_column_names['char']].values).astype(np.float32),
            np.asarray(X_pred_values[feature_column_names['word']].values).astype(np.float32),
            np.asarray(X_pred_values[feature_column_names['par']].values).astype(np.float32),
            np.asarray(X_pred_values[feature_column_names['rest']].values).astype(np.float32)
        ]
    )

    #Use LabelEncoder to convert y_pred to semantic labels
    y_pred_int = np.argmax(y_predictions, axis=1)

    labels = np.load(f"../sherlock/deploy/classes_labels.npy", allow_pickle=True)

    y_predictions = np.empty(shape=137353, dtype="U30")

    n1 = 0
    for i in labels:
        n2 = 0
        for j in y_pred_int:
            if n1 == j:
                y_predictions[n2] = i
            n2 += 1
        n1 += 1

    return y_predictions
