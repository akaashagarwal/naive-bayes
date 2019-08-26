"""Naive Bayes setup file."""
from setuptools import find_packages, setup

REQUIREMENTS = [
    'pandas == 0.25.1', 'numpy == 1.17.0', 'scikit-learn == 0.20.4'
]

setup(name="naivebayes",
      version='2.0.0',
      description="Train a Naive Bayes classifier to predict diabetes and \
    compare results with sklearn.",
      long_description=open('README.md').read(),
      author="Akaash Agarwal",
      url="https://github.com/akaashagarwal/naive-bayes",
      include_package_data=True,
      packages=find_packages(),
      install_requires=REQUIREMENTS,
      zip_safe=False,
      entry_points={"console_scripts": ["naivebayes = src.runner:main"]})
