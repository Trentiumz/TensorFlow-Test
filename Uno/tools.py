import tensorflow as tf
import matplotlib.pyplot as plt


# UNIVERSAL -------------------------------------------------------------

def get_features(dict_of_features, categoricals, numerics):
    feature_cols = []
    for feature_name in categoricals:
        vocabs = dict_of_features[feature_name].unique()
        feature_cols.append(tf.feature_column.sequence_categorical_column_with_vocabulary_list(feature_name, vocabs))
    for feature_name in numerics:
        feature_cols.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))
    return feature_cols


def get_outputs(features_dict, model):
    return [x["probabilities"] for x in
            model.predict(get_regression_input_fn(features_dict, [0] * len(list(features_dict.items())[0][1])))]

def plot_greyscale_image(the_figure):
    plt.figure()
    plt.imshow(the_figure)
    plt.grid(False)
    plt.show()

def plot_color_image(the_image):
    plt.figure()
    plt.imshow(the_image, cmap=plt.cm.binary)
    plt.show()


# LINEAR REGRESSOR ----------------------------------------------------------

def get_regression_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(
            num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


def get_incomplete_regression_input_fn(data_df, batch_size=256):
    return lambda: tf.data.Dataset.from_tensor_slices(dict(data_df)).batch(batch_size)


def make_linear_regressor(features):
    return tf.estimator.LinearClassifier(feature_columns=features)


# CLASSIFIERS --------------------------------------------------------------

def make_classifier(features, hidden_units, class_count):
    return tf.estimator.DNNClassifier(feature_columns=features, hidden_units=hidden_units, n_classes=class_count)


def get_classifier_input_fn(data_df, label_df, shuffle=True, batch_size=256):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            dataset = dataset.shuffle(1000).repeat()

        return dataset.batch(batch_size)

    return input_fn


def get_incomplete_classifier_input_fn(features, batch_size=256):
    return lambda: tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# GENERIC NEURAL NETWORK ---------------------------------------------------
# lol there's actually nothing here; it's actually strangely so simple