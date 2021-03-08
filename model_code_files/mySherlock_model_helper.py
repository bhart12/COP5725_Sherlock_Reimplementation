import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json

#from sherlock.deploy import model_helpers


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
        Categorical encodings of train labels.
    y_val_cat
        Categorical encodings of validation labels.
    """

    #Prepare categorical label encoder
    encoder = LabelEncoder()
    encoder.fit(y_train_values)

    #Save labels used in this model to .npy file
    np.save(f"../sherlock/deploy/classes_{model_name}.npy", encoder.classes_)

    # Convert train labels
    y_train_int = encoder.transform(y_train_values)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    # Convert val labels
    y_val_int = encoder.transform(y_val_values)
    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    return y_train_cat, y_val_cat


def create_sherlock_model(model_name: str, with_weights: bool):
    """Load model architecture and populate with pretrained weights.

    Parameters
    ----------
    model_name
        Identifier for retrained model.
    with_weights
        Whether to populate the model with trained weights.

    Returns
    -------
    sherlock_model
        Compiled sherlock model.
    callbacks
        Callback configuration for model retraining.
    """

    lr = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]

    #Maybe change this to model_name.json
    file = open(f"../models/sherlock_model.json", "r")
    sherlock_model = model_from_json(file.read())
    file.close()

    if with_weights:
        file = open(f"../models/{model_name}_model.json", "r")
        sherlock_model = model_from_json(file.read())
        file.close()

        sherlock_model.load_weights(f"../models/{model_name}_weights.h5")
    else:
        file = open(f"../models/sherlock_model.json", "r")
        sherlock_model = model_from_json(file.read())
        file.close()

    sherlock_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return sherlock_model, callbacks

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
        False indicates to create new model from scratch based on ../models/sherlock_model.json
        while True indicates to reopen already existing model. (Trained model must already exits if True)
    """

    if model_name == "sherlock":
        raise ValueError(
            """model_name cannot be equal to 'sherlock'
            to avoid overwriting pretrained model.
            """
        )

    feature_cols = get_processed_feature_category_dicts()
    y_train_categories, y_val_categories = _convert_list_of_labels_to_categorical_label_encodings(y_train_values, y_val_values, model_name)
    sherlock_model, callbacks = create_sherlock_model(model_name, new_model_flag)

    print("Successfully loaded and compiled model, now fitting model on data.")
    print("My file training!")


    sherlock_model.fit(
        [
            np.asarray(X_train_values[feature_cols['char']].values).astype(np.float32),
            np.asarray(X_train_values[feature_cols['word']].values).astype(np.float32),
            np.asarray(X_train_values[feature_cols['par']].values).astype(np.float32),
            np.asarray(X_train_values[feature_cols['rest']].values).astype(np.float32)
        ],
        y_train_categories,
        validation_data=(
            [
                np.asarray(X_val_values[feature_cols['char']].values).astype(np.float32),
                np.asarray(X_val_values[feature_cols['word']].values).astype(np.float32),
                np.asarray(X_val_values[feature_cols['par']].values).astype(np.float32),
                np.asarray(X_val_values[feature_cols['rest']].values).astype(np.float32)
            ],
            y_val_categories
        ),
        callbacks=callbacks, epochs=100, batch_size=256
    )


    _save_trained_model(sherlock_model, model_name)

    print('Retrained Sherlock.')
