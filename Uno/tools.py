import pandas as pd
import tensorflow as tf

# data_df is the input data, label_df is the expected output data, num_epochs is how many times a model trains for each "batch"
# It literally just returns a function which gives a random batch from our dataset
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(
            num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


def make_linear_regressor(features):
    return tf.estimator.LinearClassifier(feature_columns=features)


# dict_of_features can also be a normal pandas dataframe
def get_features(dict_of_features, categoricals, numerics):
    feature_cols = []
    for feature_name in categoricals:
        vocabs = dict_of_features[feature_name].unique()
        feature_cols.append(tf.feature_column.sequence_categorical_column_with_vocabulary_list(feature_name, vocabs))
    for feature_name in numerics:
        feature_cols.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))
    return feature_cols


# feature_lists = {feature_name : [list of inputs of size x], ...}, label_lists=[list of outputs and size x]
def make_input_fn_from_basic(features_dict, label_lists, num_epochs=10, shuffle=True, batch_size=32):
    label_series = pd.Series(data=label_lists)
    features = pd.DataFrame.from_dict(features_dict)
    return make_input_fn(features, label_series, num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size)


def get_outputs(features_dict, model):
    return [x["probabilities"] for x in
            model.predict(make_input_fn_from_basic(features_dict, [0] * len(list(features_dict.items())[0][1])))]
