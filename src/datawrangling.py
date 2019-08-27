"""Module to load and preprocess data."""
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


class DataWrangling():
    """Class for data loading and preprocessing.

    Attributes:
        data_path (str): Path to data file.

    """

    def __init__(self, data_path: str = '') -> None:
        """Initialise.

        Args:
            data_path (str, optional): Description

        """
        self.data_path = data_path
        self._data = pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        """Return deep copy of self._data.

        Returns:
            pd.DataFrame

        """
        return self._data.copy(deep=True)

    def load_data(self) -> None:
        """Read data from file given by self.data_path."""
        self._data = pd.read_csv(self.data_path)

    @staticmethod
    def _find_non_zero_mean(data_frame: pd.DataFrame, feature: str,
                            class_column: str, class_val: Any) -> float:
        """Find mean of non zero values of feature column with given class_val.

        Args:
            data_frame (pd.DataFrame)
            feature (str): Name of data frame feature column.
            class_column (str): Name of class/response column in data frame.
            class_val (Any): Specific class value for which calculation will be
                             done.

        Returns:
            float

        """
        # Find the mean of the non-zero values of given feature with given
        # class_val.
        idx_class_val = (data_frame[class_column] == class_val)
        idx_feature_non_zero = (data_frame[feature] != 0)
        df_non_zero = data_frame[idx_class_val & idx_feature_non_zero]
        feature_non_zero_mean = df_non_zero[feature].mean()
        return feature_non_zero_mean

    def _fill_null_val_column(self, data_frame: pd.DataFrame, feature: str,
                              class_column: str) -> pd.DataFrame:
        """Calculate mean of nonzero feature values and use that to fill zeros.

        In particular, this function first calculates the mean of the non-zero
        values of the given feature for a particular class, and then uses that
        to overwrite the zeros in the vector of that feature for that class.

        Args:
            data_frame (pd.DataFrame)
            feature (str): Name of data frame feature column.
            class_column (str): Name of class/response column in data frame.

        Returns:
            pd.DataFrame

        """
        unique_class_vals = data_frame[class_column].unique()
        for class_val in unique_class_vals:

            feature_non_zero_mean = self._find_non_zero_mean(
                data_frame=data_frame,
                feature=feature,
                class_column=class_column,
                class_val=class_val)

            # Assign the mean calculated in the previous statement to the zero
            # values of the given feature with the given class_val.
            idx_feature_zero = (data_frame[feature] == 0)
            idx_class_val = (data_frame[class_column] == class_val)
            data_frame.loc[idx_feature_zero
                           & idx_class_val, feature] = feature_non_zero_mean
        return data_frame

    def fill_nul_vals(self, data_frame: pd.DataFrame, class_column: str,
                      features: List[str]) -> pd.DataFrame:
        """Fill zero values in columns of given features.

        This function takes a set of features for which it will attempt to fill
        in the zeros in the vectors of the features.

        Args:
            data_frame (pd.DataFrame)
            class_column (str): Name of class/response column in data frame.
            features (List[str]): List of feature names for which null values
                                  need to be filled.

        Returns:
            pd.DataFrame

        """
        new_data_frame = data_frame.copy(deep=True)
        for feature in features:
            new_data_frame = self._fill_null_val_column(
                data_frame=new_data_frame,
                feature=feature,
                class_column=class_column)
        return new_data_frame

    @staticmethod
    def discretize_features(data_frame: pd.DataFrame, features: List[str],
                            n_bins: int) -> np.array:
        """Create a one-hot encoding after binning each given feature column.

        Args:
            data_frame (pd.DataFrame)
            features (List[str]): Set of feature columns.
            n_bins (int): The number of bins needed to discretize the feature.

        Returns:
            np.array: One-hot encoded feature vectors after binning.

        """
        est = KBinsDiscretizer(n_bins=n_bins,
                               encode='onehot-dense',
                               strategy='quantile')
        array_form = np.array(data_frame[features])
        if len(features) == 1:
            array_form = array_form.reshape(-1, 1)
        est.fit(array_form)
        return est.transform(array_form)

    @staticmethod
    def split_train_test(x_data: np.array, y_labels: np.array, features: list,
                         test_size: float) -> dict:
        """Split into training and test sets by proportion given by test_size.

        Returns:
            dict: Dictionary with training data, training labels, test data,
                  and test labels.

        """
        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_labels,
            test_size=test_size,
            random_state=41,
            shuffle=True,
            stratify=y_labels)
        return_val = {
            'x_train': pd.DataFrame(x_train, columns=features),
            'x_test': pd.DataFrame(x_test, columns=features),
            'y_train': pd.Series(y_train),
            'y_test': pd.Series(y_test)
        }
        return return_val
