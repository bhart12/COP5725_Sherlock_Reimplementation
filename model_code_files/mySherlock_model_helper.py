import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json



def create_sherlock_model(model_name: str, with_weights: bool):
    """Load model architecture and populate with pretrained weights.

    Parameters
    ----------
    model_name
        Identifier for retrained model.
    with_weights
        Whether to populate the model with trained weights, or start with new untrained model.

    Returns
    -------
    sherlock_model
        Compiled sherlock model.
    callbacks
        Callback configuration for model retraining.
    """

    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    #Declare model variable
    sherlock_model = None

    if with_weights:
        #Open previously trained model
        file = open(f"../models/{model_name}_model.json", "r")
        sherlock_model = model_from_json(file.read())
        file.close()

        sherlock_model.load_weights(f"../models/{model_name}_weights.h5")
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

        sherlock_model = keras.Model(inputs=[input1, input2, input3, input4], outputs=output, name="Sherlock_Reimplementation_Model")


        #opt = keras.optimizers.Adam(learning_rate=0.0001)
        #model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    #print(sherlock_model.summary())

    sherlock_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return sherlock_model, callbacks

def _convert_list_of_labels_to_categorical_label_encodings(y_train_values, y_val_values, model_name) -> (list, list):
    """Encode semantic type string labels as categoricals.

    Parameters
    ----------
    y_train_values
        Train labels.
    y_val_values
        Validation labels.
    model_name
        Identifier of retrained model.

    Returns
    -------
    y_train_cat
        Categorical encodings (binary matrix representation) of train labels.
    y_val_cat
        Categorical encodings (binary matrix representation) of validation labels.
    """

    #Prepare categorical label encoder
    encoder = LabelEncoder()
    encoder.fit(y_train_values)

    #Save labels used in this model to .npy file
    np.save(f"../sherlock/deploy/classes_{model_name}.npy", encoder.classes_)

    #Convert train label strings to normalized encoding
    #y_train_int = encoder.fit_transform(y_train_values)
    y_train_int = encoder.transform(y_train_values)
    #Convert normalized encoding to binary matrix representation of the input
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    #Convert validation label strings to normalized encoding
    y_val_int = encoder.transform(y_val_values)
    #Convert normalized encoding to binary matrix representation of the input
    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    return y_train_cat, y_val_cat


def get_processed_feature_category_dicts() -> dict:
    """Get feature identifiers per feature set, to map features to feature sets.

    Returns
    -------
    feature_cols_dict
        Dictionary with lists of feature identifiers per feature set.
    """
    feature_cols_dict = {}
    for feature_set in ['char', 'word', 'par', 'rest']:
        feature_cols_dict[feature_set] = pd.read_csv(
            f"../sherlock/features/feature_column_identifiers/{feature_set}_col.tsv",
            sep='\t', index_col=0, header=None, squeeze=True,
        ).to_list()
    return feature_cols_dict

def _save_trained_model(sherlock_model, model_name: str):
    """Save weights of retrained sherlock model.

    Parameters
    ----------
    sherlock_model
        Retrained sherlock model.
    model_name
        Identifier for retrained model.
    """

    model_json = sherlock_model.to_json()
    with open(f"../models/{model_name}_model.json", "w") as json:
        json.write(model_json)

    sherlock_model.save_weights(f"../models/{model_name}_weights.h5")

def train_sherlock(
    X_train_values: pd.DataFrame,
    y_train_values: list,
    X_val_values: pd.DataFrame,
    y_val_values: list,
    model_name: str,
    new_model_flag: bool,
):
    """Train weights of sherlock model from existing NN architecture.

    Parameters
    ----------
    X_train_values
        Train data to train model on.
    y_train_values
        Train labels to train model with.
    X_val_values
        Validation data to steer early stopping.
    y_val_values
        Validation labels.
    model_name
        Identifier for retrained model.
    new_model_flag
        False indicates to create new model from scratch
        while True indicates to reopen already existing model. (Trained model must already exits if True)
    """

    if model_name == "sherlock":
        raise ValueError(
            """model_name cannot be equal to 'sherlock'
            to avoid overwriting pretrained model.
            """
        )

    feature_column_names = get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = _convert_list_of_labels_to_categorical_label_encodings(y_train_values, y_val_values, model_name)
    sherlock_model, callbacks = create_sherlock_model(model_name, new_model_flag)

    print("Successfully loaded and compiled model, now fitting model on data.")
    print("My file training!")


    sherlock_model.fit(
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


    _save_trained_model(sherlock_model, model_name)

    print('Retrained Sherlock.')


def predict_sherlock(X_pred_values: pd.DataFrame, model_name: str) -> np.array:
    """Use sherlock model to generate predictions for X.

    Parameters
    ----------
    X_pred_values
        Test data to to get predictions for
    model_name
        Identifier of a trained model to use for generating predictions.

    Returns
    -------
    Array with predictions for X.
    """
    #Get model based off of model_name
    sherlock_model, callbacks = create_sherlock_model(model_name, True)
    #Get feature names by category in dict
    feature_column_names = get_processed_feature_category_dicts()

    #Predict labels
    y_predictions = sherlock_model.predict(
        [
            np.asarray(X_pred_values[feature_column_names['char']].values).astype(np.float32),
            np.asarray(X_pred_values[feature_column_names['word']].values).astype(np.float32),
            np.asarray(X_pred_values[feature_column_names['par']].values).astype(np.float32),
            np.asarray(X_pred_values[feature_column_names['rest']].values).astype(np.float32)
        ]
    )

    #Use LabelEncoder to convert y_pred to semantic labels
    y_pred_int = np.argmax(y_predictions, axis=1)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        f"../sherlock/deploy/classes_{model_name}.npy",
        allow_pickle=True
    )
    y_predictions = encoder.inverse_transform(y_pred_int)

    return y_predictions
