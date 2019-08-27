"""Module with Naive-Bayes classifier."""
from functools import reduce
from typing import List, Union

import pandas as pd


class NaiveBayes():
    """Naive Bayes classifier class.

    Attributes:
        class_priors (dict): Prior probabilities of classes based on train set.
        data_store (dict): Contains training and test sets as well as labels.
        features (list): List of all features of input.
        likelihoods (dict): Will store likelihood of each feature per class.

    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Initialise class attributes.

        Args:
            x_train (pd.DataFrame): Input data frame.
            y_train (pd.Series): Class labels for input data frame.

        """
        self.features = list(x_train.columns)
        self.data_store = {'y_train': y_train, 'x_train': x_train}
        self.likelihoods: dict = {}
        self.class_priors: dict = {}
        for feature in self.features:
            self.likelihoods[feature] = {}
            for class_val in self.data_store['y_train'].unique():
                self.likelihoods[feature].update({class_val: 0})
                self.class_priors.update({class_val: 0})

    def _find_feature_likelihoods(self, class_val: Union[int, float],
                                  feature: str) -> float:
        """Find likelihood of given feature for given class.

        Args:
            class_val (Union[int, float]): Class label.
            feature (str): Feature name.

        Returns:
            float

        """
        feature_series = self.data_store['x_train'][feature]
        class_val_idx = self.data_store['y_train'] == class_val
        feature_series_class = feature_series[class_val_idx]
        feature_mean_class = feature_series_class.mean()
        return feature_mean_class

    def _find_likelihoods(self) -> None:
        """Find likelihood for all features for each class."""
        for feature in self.features:
            for class_val in self.data_store['y_train'].unique():
                feature_likelihood_class = self._find_feature_likelihoods(
                    feature=feature, class_val=class_val)
                self.likelihoods[feature][class_val] = feature_likelihood_class

    def _find_class_priors(self) -> None:
        """Find class priors for given training set."""
        total_dataset_size = self.data_store['y_train'].shape[0]
        for class_val in self.data_store['y_train'].unique():
            class_val_count = sum(self.data_store['y_train'] == class_val)
            self.class_priors[class_val] = class_val_count / total_dataset_size

    def fit(self) -> None:
        """Fit naive Bayes model to given training data.

        This consists of finding p(x|C) and p(C), where x is an input feature
        and C is a class label, for all x over all C.

        """
        self._find_class_priors()
        self._find_likelihoods()

    def _find_probability(self, feature: str, val: int,
                          class_val: Union[int, float]) -> float:
        """Return probability of feature having given val for given class_val.

        This returns the likelihood of the feature having a one-hot encoded
        value of val for the given class label class_val.

        If the value is 1, then the probability stored in self.likelihoods of
        the given feature for the given class is returned. Else, one minus that
        probability is returned (as the likelihood for the feature was calcula-
        -ted when val = 1, the probability for val = 0 is simply derived from
        subtracting the probability for val = 1 from 1).

        Args:
            feature (str): Feature name.
            val (int): The one-hot encoded feature value of given input.
            class_val (Union[int, float]): Class label, can be int or float.

        Returns:
            float: Probability of feature taking on value val for given class.

        """
        probability = self.likelihoods[feature][class_val]
        return probability if val else (1 - probability)

    def _find_feature_probs(self, x_test_instance: pd.Series,
                            class_val: Union[float, int]) -> List[float]:
        """Find the probabilites of the features having given values for class.

        Args:
            x_test_instance (pd.Series): Input example.
            class_val (Union[float, int]): Class label for given input.

        Returns:
            List[float]

        """
        feature_probs = []
        for feature, val in x_test_instance.items():
            feature_prob = self._find_probability(feature, val, class_val)
            feature_probs.append(feature_prob)
        return feature_probs

    def _predict_for_input(self,
                           x_test_instance: pd.Series) -> Union[int, float]:
        """Predict most likely class label for given input.

        In particular, the decision rule used here in conjunction with the tra-
        ined naive Bayes model is the maximum a posteriori decision rule.

        The corresponding classifier, the Bayes classifier, assigns the class
        y_hat for a given input vector X as y_hat = argmax_C p(C)p(X|C) over
        all C, where C is the set of all possible class labels.

        Intuitively, this means that the classifier chooses that class label
        which has the highest probability to have generated the given input X.

        Args:
            x_test_instance (pd.Series): Input vector.

        Returns:
            Union[int, float]: Class label.

        """
        def product(x_1, x_2) -> float:
            """Return product of x_1 and x_2.

            Args:
                x_1 (float)
                x_2 (flpat)

            Returns:
                float

            """
            return x_1 * x_2

        max_posterior_prob = 0

        # Calculate posterior probabilties over class label set C and find max.
        for class_val in self.data_store['y_train'].unique():

            # Find all p(x|class_val) where x is a feature in the input set X.
            feature_probs = self._find_feature_probs(x_test_instance,
                                                     class_val)

            # Find likelihood of data coming from class class_val and also the
            # prior probability of class_val.
            likelihood = reduce(product, feature_probs)
            prior = self.class_priors[class_val]

            # Use posterior = prior * likelihood to find posterior probability
            # of the input X coming from class class_val.
            posterior = prior * likelihood

            # Keep track of class with highest posterior probability.
            if posterior > max_posterior_prob:
                max_posterior_prob = posterior
                max_posterior_class = class_val

        return max_posterior_class

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        """Predict class labels for given set of input vectors x_test.

        Args:
            x_test (pd.DataFrame): Input vector.

        Returns:
            pd.Series: Predictions for given input matrix.

        """
        predictions = []
        for _, x_test_instance in x_test.iterrows():
            prediction = self._predict_for_input(
                x_test_instance=x_test_instance)
            predictions.append(prediction)
        return pd.Series(predictions)
