"""Test module for naivebayes.py.

Attributes:
    TEST_DATA_PATH (str): Path to csv file with data.

"""
import pytest
import numpy as np  # noqa: I100
import pandas as pd

from sklearn.naive_bayes import BernoulliNB  # noqa: I202
from src.algorithm import NaiveBayes
from src.datawrangling import DataWrangling

TEST_DATA_PATH = 'src/data/data.csv'


def get_data(n_bins: int) -> dict:
    """Retrieve data and perform preprocessing steps and return result.

    Args:
        n_bins (int): Number of bins to discretize features into.

    Returns:
        dict: Dictionary containing test and training sets and labels.

    """
    dframe = pd.read_csv(TEST_DATA_PATH)
    obj1 = DataWrangling(data_path=TEST_DATA_PATH)
    obj1.load_data()
    features = [
        'Pglucose', 'Preg', 'Dbp', 'BMI', 'Dpfunc', 'Age', 'Insulin', 'Skin'
    ]

    input_df = obj1.data[features + ['Response']].astype(np.float64)
    no_zero_vals = obj1.fill_nul_vals(data_frame=input_df,
                                      class_column='Response',
                                      features=features)
    new_feature_list = []
    for feature in features:
        for bin_val in range(1, n_bins + 1):
            new_feature_list.append(feature + str(bin_val))
    discr_arr = obj1.discretize_features(data_frame=no_zero_vals,
                                         features=features,
                                         n_bins=n_bins)
    discr_df = pd.DataFrame(discr_arr, columns=new_feature_list)
    discr_df['Response'] = dframe['Response']
    splits = obj1.split_train_test(
        x_data=discr_df[new_feature_list].to_numpy(),
        y_labels=discr_df['Response'].to_numpy(),
        features=new_feature_list,
        test_size=0.25)
    return splits


@pytest.mark.parametrize("n_bins", [2, 4])
def test_integration_naive_bayes(n_bins: int) -> None:
    """Test full Naive Bayes classifier prediction vs scikit-learn Naive Bayes.

    Args:
        n_bins (int): Number of bins to discretize features into.

    """
    splits = get_data(n_bins)
    obj = NaiveBayes(x_train=splits['x_train'], y_train=splits['y_train'])
    obj.fit()
    prediction_score = sum(obj.predict(splits['x_test']) ==  # noqa: W504
                           splits['y_test']) / splits['y_test'].shape[0]

    bnb = BernoulliNB()

    x_train = splits['x_train']
    y_train = splits['y_train']
    x_test = splits['x_test']
    y_test = splits['y_test']
    pred = bnb.fit(x_train, y_train).predict(x_test)
    sklearn_score = sum(pd.Series(pred) == y_test) / y_test.shape[0]

    assert prediction_score == sklearn_score
