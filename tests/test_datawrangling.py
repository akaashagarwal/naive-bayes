"""Test module for src/datawrangling.py."""
import numpy as np
import pandas as pd

from src.datawrangling import DataWrangling  # noqa: I202


def test_load_data():
    """Test load_data()."""
    obj = DataWrangling(data_path='src/data/data.csv')
    obj.load_data()
    assert all(obj.data.columns == pd.read_csv('src/data/data.csv').columns)


def test_fill_nul_vals():
    """Test fill_null_vals()."""
    recs = [{
        'test_feature_1': 0,
        'test_feature_2': 0,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 5,
        'test_class': 1
    }, {
        'test_feature_1': 2,
        'test_feature_2': 0,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 3,
        'test_class': 0
    }, {
        'test_feature_1': 0,
        'test_feature_2': 2,
        'test_class': 0
    }]
    test_df = pd.DataFrame(recs)

    obj = DataWrangling()
    return_val = obj.fill_nul_vals(
        data_frame=test_df,
        class_column='test_class',
        features=['test_feature_1', 'test_feature_2'])

    recs = [{
        'test_feature_1': 1.5,
        'test_feature_2': 5.0,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 5,
        'test_class': 1
    }, {
        'test_feature_1': 2,
        'test_feature_2': 5.0,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 3,
        'test_class': 0
    }, {
        'test_feature_1': 1.0,
        'test_feature_2': 2,
        'test_class': 0
    }]
    expected_val = pd.DataFrame(recs)

    assert return_val.equals(expected_val)


def test_discretize_features_one_feature():
    """Test discretize_features for just one feature."""
    recs = [{
        'test_feature_1': 1,
        'test_feature_2': 1,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 1,
        'test_class': 1
    }, {
        'test_feature_1': 3,
        'test_feature_2': 3,
        'test_class': 1
    }, {
        'test_feature_1': 5,
        'test_feature_2': 5,
        'test_class': 0
    }, {
        'test_feature_1': 5,
        'test_feature_2': 5,
        'test_class': 0
    }]
    test_df = pd.DataFrame(recs)
    obj = DataWrangling()
    return_val = obj.discretize_features(data_frame=test_df,
                                         features=['test_feature_1'],
                                         n_bins=3)

    expected_val = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [0., 0., 1.]])

    assert np.array_equal(return_val, expected_val)


def test_discretize_features_two_features():
    """Test discretize_features for more than one feature."""
    recs = [{
        'test_feature_1': 1,
        'test_feature_2': 1,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 1,
        'test_class': 1
    }, {
        'test_feature_1': 3,
        'test_feature_2': 3,
        'test_class': 1
    }, {
        'test_feature_1': 5,
        'test_feature_2': 5,
        'test_class': 0
    }, {
        'test_feature_1': 5,
        'test_feature_2': 5,
        'test_class': 0
    }]
    test_df = pd.DataFrame(recs)
    obj = DataWrangling()
    return_val = obj.discretize_features(
        data_frame=test_df,
        features=['test_feature_1', 'test_feature_2'],
        n_bins=3)

    expected_val = np.array([[1., 0., 0., 1., 0.,
                              0.], [1., 0., 0., 1., 0., 0.],
                             [0., 1., 0., 0., 1.,
                              0.], [0., 0., 1., 0., 0., 1.],
                             [0., 0., 1., 0., 0., 1.]])

    assert np.array_equal(return_val, expected_val)


def test_split_train_test():
    """Test split_train_test()."""
    recs = [{
        'test_feature_1': 1,
        'test_feature_2': 1,
        'test_class': 1
    }, {
        'test_feature_1': 1,
        'test_feature_2': 1,
        'test_class': 1
    }, {
        'test_feature_1': 3,
        'test_feature_2': 3,
        'test_class': 1
    }, {
        'test_feature_1': 5,
        'test_feature_2': 5,
        'test_class': 0
    }, {
        'test_feature_1': 5,
        'test_feature_2': 5,
        'test_class': 0
    }]
    test_df = pd.DataFrame(recs)
    features = ['test_feature_1', 'test_feature_2']
    test_size = 0.33
    class_col = 'test_class'
    obj = DataWrangling()
    return_val = obj.split_train_test(x_data=test_df[features].to_numpy(),
                                      y_labels=test_df[class_col].to_numpy(),
                                      features=features,
                                      test_size=test_size)
    expected_keys = ['x_train', 'y_train', 'x_test', 'y_test']
    assert all(key in return_val for key in expected_keys)
    assert all(
        isinstance(val, (pd.Series, pd.DataFrame))
        for _, val in return_val.items())
