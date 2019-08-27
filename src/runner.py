"""Module to run both custom-developed Naive Bayes and SKLearn to compare.

Attributes:
    DATA_PATH (str): Path to the .csv file containing the dataset.

"""
# pylint: disable=R0913
from typing import List

import pandas as pd
from sklearn.naive_bayes import BernoulliNB  # noqa: I202

from .algorithm import NaiveBayes
from .datawrangling import DataWrangling

DATA_PATH = 'src/data/data.csv'


def get_sklearn_prediction_score(data_splits: dict) -> float:
    """Get prediction accuracy of SKLearn's Naive Bayes classifier on test set.

    This function will fit a Naive Bayes model, predict on the test set, and r-
    -eturn the accuracy of the predictions.

    Args:
        data_splits (dict): Dictionary of training and test data and labels.

    Returns:
        float: Prediction accuracy on test set.

    """
    bnb = BernoulliNB()
    bnb.fit(data_splits['x_train'], data_splits['y_train'])
    predictions = bnb.predict(data_splits['x_test'])
    correct_predictions = sum(pd.Series(predictions) == data_splits['y_test'])
    return correct_predictions / data_splits['y_test'].shape[0]


def get_prediction_score(data_splits: dict) -> float:
    """Get prediction accuracy of custom Naive Bayes classifier on test set.

    This function will fit a Naive Bayes model, predict on the test set, and r-
    -eturn the accuracy of the predictions.

    Args:
        data_splits (dict): Dictionary of training and test data and labels.

    Returns:
        float: Prediction accuracy on test set.

    """
    obj = NaiveBayes(x_train=data_splits['x_train'],
                     y_train=data_splits['y_train'])
    obj.fit()
    predictions = obj.predict(data_splits['x_test'])
    correct_predictions = sum(predictions == data_splits['y_test'])
    return correct_predictions / data_splits['y_test'].shape[0]


def get_feature_list(n_bins: int, features: List[str]) -> List[str]:
    """Return list of features where each feature is discretized into buckets.

    For eg., if a feature 'glucose' has been discretized into 3 bins, then this
    function will return ['glucose1', 'glucose2', 'glucose3'].

    Args:
        n_bins (int): Number of bins each feature has been discretized into.
        features (List[str]): List of features that have been discretized in
                              buckets.

    Returns:
        List[str]

    """
    new_feature_list = []
    for feature in features:
        for bin_val in range(1, n_bins + 1):
            new_feature_list.append(feature + str(bin_val))
    return new_feature_list


def preprocess_data(obj: DataWrangling, data: pd.DataFrame,
                    features: List[str], no_zero_features: List[str],
                    class_col: str, n_bins: int, test_size: float) -> dict:
    """Fill zero values, discretize features, and split into train/test set.

    Args:
        obj (DataWrangling): DataWrangling object.
        data (pd.DataFrame): Full dataset, containing test and train data.
        features (List[str]): List of features of dataset.
        no_zero_features (List[str]): List of features that don't need zero
                                      value filling.
        class_col (str): Label column name.
        n_bins (int): Number of bins to discretize the features.
        test_size (float): Proportion of data to use as test set.

    Returns:
        dict: Dictionary containing keys for training and test data, as well as
              their labels.

    """
    input_df = data.copy(deep=True)

    # Fill zero/missing values for features.
    input_df = obj.fill_nul_vals(data_frame=input_df,
                                 class_column=class_col,
                                 features=[
                                     feature for feature in features
                                     if feature not in no_zero_features
                                 ])

    # Bucket continuous features into n_bins.
    discr_arr = obj.discretize_features(data_frame=input_df,
                                        features=features,
                                        n_bins=n_bins)
    new_feature_list = get_feature_list(n_bins=n_bins, features=features)
    discr_df = pd.DataFrame(discr_arr, columns=new_feature_list)
    discr_df[class_col] = data[class_col]

    # Split dataset into train and test set.
    data_splits = obj.split_train_test(
        x_data=discr_df[new_feature_list].to_numpy(),
        y_labels=discr_df[class_col].to_numpy(),
        features=new_feature_list,
        test_size=test_size)

    return data_splits


def main() -> None:
    """Run performance comparison between custom and sklearn's Naive Bayes."""
    obj = DataWrangling(data_path=DATA_PATH)
    obj.load_data()

    # Set parameters.
    data = obj.data
    class_col = 'Response'
    features = list(data.columns)
    features.remove(class_col)
    no_zero_features = ['Preg']
    n_bins = 5
    test_size = 0.25

    # Fill zero values where applicable, discretize continuous features, split
    # into train and test set.
    data_splits = preprocess_data(obj=obj,
                                  data=data,
                                  features=features,
                                  no_zero_features=no_zero_features,
                                  class_col=class_col,
                                  n_bins=n_bins,
                                  test_size=test_size)

    # Make predictions on test set using custom Naive Bayes and sklean's Naive
    # Bayes, and print results.
    prediction_score = get_prediction_score(data_splits)
    sklearn_score = get_sklearn_prediction_score(data_splits)

    print(
        'Performance of Bernoulli Naive Bayes classifier built from scratch: ',
        prediction_score)
    print('Performance of scikit-learn\'s Bernoulli Naive Bayes classifier: ',
          sklearn_score)


if __name__ == '__main__':
    main()
