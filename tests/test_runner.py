"""Test module for src/runner.py."""

# pylint: disable=C0103, W0603, C0111
# flake8: noqa=D103

import pandas as pd
from src.datawrangling import DataWrangling
from src.runner import (
    get_feature_list,
    get_prediction_score,
    get_sklearn_prediction_score,
    main,
    preprocess_data,
)

recs = [{
    'feature_1': 1,
    'feature_2': 4,
    'label': 1
}, {
    'feature_1': 2,
    'feature_2': 5,
    'label': 1
}, {
    'feature_1': 3,
    'feature_2': 6,
    'label': 1
}, {
    'feature_1': 4,
    'feature_2': 1,
    'label': 0
}, {
    'feature_1': 5,
    'feature_2': 2,
    'label': 0
}, {
    'feature_1': 6,
    'feature_2': 3,
    'label': 0
}]

data = pd.DataFrame(recs)
class_col = 'label'
features = list(data.columns)
features.remove(class_col)
no_zero_features = ['Preg']
n_bins = 3
test_size = 0.33


def test_preprocess_data():
    global data, features, class_col, n_bins, test_size, no_zero_features
    obj = DataWrangling()
    return_val = preprocess_data(obj=obj,
                                 data=data,
                                 features=features,
                                 no_zero_features=no_zero_features,
                                 class_col=class_col,
                                 n_bins=n_bins,
                                 test_size=test_size)

    expected_keys = ['x_train', 'y_train', 'x_test', 'y_test']
    assert all(key in return_val for key in expected_keys)
    assert all(
        isinstance(val, (pd.Series, pd.DataFrame))
        for _, val in return_val.items())

    assert return_val['x_train'].shape[0] == 4
    assert return_val['x_test'].shape[0] == 2
    assert return_val['x_train'].shape[1] == 6


def test_prediction_score():
    global data, features, class_col, n_bins, test_size, no_zero_features
    obj = DataWrangling()
    splits = preprocess_data(obj=obj,
                             data=data,
                             features=features,
                             no_zero_features=no_zero_features,
                             class_col=class_col,
                             n_bins=n_bins,
                             test_size=test_size)
    prediction_score = get_prediction_score(splits)
    splits = splits['x_train']
    assert isinstance(prediction_score, float)
    assert prediction_score >= 0.0
    assert prediction_score <= 1.0


def test_sklearn_prediction_score():
    global data, features, class_col, n_bins, test_size, no_zero_features
    obj = DataWrangling()
    splits = preprocess_data(obj=obj,
                             data=data,
                             features=features,
                             no_zero_features=no_zero_features,
                             class_col=class_col,
                             n_bins=n_bins,
                             test_size=test_size)
    prediction_score = get_sklearn_prediction_score(splits)
    splits = splits['x_train']
    assert isinstance(prediction_score, float)
    assert prediction_score >= 0.0
    assert prediction_score <= 1.0


def test_get_feature_list():
    return_val = get_feature_list(3, ['feature_'])
    assert return_val == ['feature_1', 'feature_2', 'feature_3']


def test_main():
    assert main() is None
