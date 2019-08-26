# Summary

This package contains a Bernoulli Naive Bayes classifier written from scratch and trained/tested on a dataset to predict the onset of diabetes. To check the correctness of the implemented algorithm, scikit-learn's Bernoulli Naive Bayes classifier is also trained on the same training set and tested on the same test set. Its resulting performance is compared with that of the custom built Naive Bayes classifier.

## Description
The primary objective of this project was to accurately translate the mathematics behind the Bernoulli Naive Bayes classifier into code.

Hence, the focus here is not to maximise the prediction accuracy as such, and therefore steps to visualize the data and perform data exploration and analysis have been skipped.

Minimal preprocessing in the form of filling zero values for features where it doesn't make sense to have zero values has been done. Additionally, continuous features have been discretized so as to obtain a one-hot encoding of them.

Finally, to test whether the implemented Naive Bayes algorithm is working as expected, its performance on the test set is compared with that of the Bernoulli Naive Bayes classifier found in scikit-learn.

## Naive Bayes

For this project, we use the *maximum a posteriori* decision rule in conjunction with the naive Bayes probability model, to give us the Bayes classifer.

More: <https://en.wikipedia.org/wiki/Naive_Bayes_classifier> 

## Data

The data set used is one which contains information on a set of patients who are either diabetic or non-diabetic, given a set of predictors. The goal is to build a classifier and make a prediction on whether a patient is diabetic or not, given the predictors.

#### Predictors:

- **Pregnancies**: Number of times pregnant
- **GlucosePlasma**: glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressureDiastolic**: Diastolic blood pressure (mm Hg)
- **SkinThicknessTriceps**: skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)

#### Response:

- **Response**: Class variable (0 for non-diabetic or 1 for diabetic); 268 of 768 are 1; the others are 0

Ref: <https://www.kaggle.com/uciml/pima-indians-diabetes-database>

## Requirements

- Python 3.7.4
- Python dependencies:
    - [Tox](https://tox.readthedocs.io/en/latest/) (optional; needed for testing)

## Installation

1. Clone this repo
2. Move into the cloned directory.

    `cd naive-bayes`

3. Create a new virtual environment for Python 3.7.4 (if not installed -> <https://virtualenv.pypa.io/en/latest/>).

    `virtualenv venv3`

4. Activate the virtual environment in the terminal/cmd.
    
    Windows:
    
    `venv3\Scripts\activate`

    Linux:

    `source venv3/bin/activate`

5. Install the package

    `python setup.py install`

## Usage

Assuming you've installed the package, simply run the following in the terminal:

`naivebayes`

This will execute the `main()` function in `src/runner.py`.

The following should be the resultant output:

```
Performance of Bernoulli Naive Bayes classifier built from scratch:  0.8645833
Performance of scikit-learn's Bernoulli Naive Bayes classifier:  0.859375
```

## Testing

There are 3 types of automated testing provisioned for this package:

1. Unit/integration tests
2. Pylint
3. Flake8

All of these will require that you have `tox` installed in your system.

`pip install tox`

### Unit/Integration Tests

While in the cloned package directory, simply run the following in a terminal/cmd:

`tox`

It will produce the following:

```
tests/test_algorithm.py::test_integration_naive_bayes[2] PASSED                                                                                                                                            [  8%]
tests/test_algorithm.py::test_integration_naive_bayes[4] PASSED                                                                                                                                            [ 16%]
tests/test_datawrangling.py::test_load_data PASSED                                                                                                                                                         [ 25%]
tests/test_datawrangling.py::test_fill_nul_vals PASSED                                                                                                                                                     [ 33%]
tests/test_datawrangling.py::test_discretize_features_one_feature PASSED                                                                                                                                   [ 41%]
tests/test_datawrangling.py::test_discretize_features_two_features PASSED                                                                                                                                  [ 50%]
tests/test_datawrangling.py::test_split_train_test PASSED                                                                                                                                                  [ 58%]
tests/test_runner.py::test_preprocess_data PASSED                                                                                                                                                          [ 66%]
tests/test_runner.py::test_prediction_score PASSED                                                                                                                                                         [ 75%]
tests/test_runner.py::test_sklearn_prediction_score PASSED                                                                                                                                                 [ 83%]
tests/test_runner.py::test_get_feature_list PASSED                                                                                                                                                         [ 91%]
tests/test_runner.py::test_main PASSED                                                                                                                                                                     [100%]

----------- coverage: platform win32, python 3.7.4-final-0 -----------
Name                   Stmts   Miss Branch BrPart  Cover   Missing
------------------------------------------------------------------
src\__init__.py            0      0      0      0   100%
src\algorithm.py          62      0     18      0   100%
src\datawrangling.py      47      0      6      0   100%
src\runner.py             48      0      6      0   100%
------------------------------------------------------------------
TOTAL                    157      0     30      0   100%

Required test coverage of 90% reached. Total coverage: 100.00%

============================================================================================== 12 passed in 2.82s ===============================================================================================
____________________________________________________________________________________________________ summary ____________________________________________________________________________________________________
  py37: commands succeeded
  congratulations :)
```

### Pylint

While in the cloned package directory, run the following:

`tox -e pylint`

Which should give a comprehensive report, ending with:

```
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```

### Flake8

While in the cloned package directory, run the following:

`tox -e flake8`

## Author
- Akaash Agarwal