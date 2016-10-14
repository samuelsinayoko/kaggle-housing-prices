"""An ML library to help with Kaggle problems.
"""
import pandas as pd
import sklearn.base



class Regressor(sklearn.base.BaseEstimator):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.model = self.estimator(X, y).fit()

    def predict(self, X):
        return self.model.predict(X)



class DataSet:
    """Helper class to manipulate the training and test datasets seamlessly.

    Attributes
    ----------
    df : dataframe
         Full data containing both the training and test datasets.
    train: dataframe
         The training dataset, kept in sync with df.
    test: dataframe
         The test dataset, kept in sync with df.
    """
    def __init__(self, raw_train, raw_test):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.train = self.raw_train.copy()
        self.test = self.raw_test.copy()
        self.df = self.merge(self.raw_train, self.raw_test)

    @staticmethod
    def merge(train, test):
        return pd.concat([train, test], axis=0, ignore_index=True)

    def split(self, alldf):
        n = self.train.shape[0]
        train = alldf.iloc[:n, :].set_index(self.raw_train.index)
        test = alldf.iloc[n:, :].set_index(self.raw_test.index)
        return train, test

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, dataframe):
        self._df = dataframe
        # Update the train and test datasets
        self.train, self.test = self.split(self._df)

    def copy(self):
        """Return a copy of the dataset."""
        ds = DataSet(self.train, self.test)
        ds.raw_train = self.raw_train
        ds.raw_test = self.raw_test
        return ds

    def apply(self, func, inplace=False):
        """Apply a function func: dataframe -> dataframe
        to the dataset and return the transformed dataset.
        Leave raw data unchanged.
        """
        df = func(self.df)
        if inplace:
            self.df = df
            return self
        else:
            ds = self.copy()
            ds.df = df
            return ds

    def __getattr__(self, attr):
        """Try to get the attribute from the the class,
        otherwise try to get it from the underlying dataframe.
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            try:
                return self.df.__getattr__(attr)
            except AttributeError:
                print("Unable to find attribute {!r} in self nor in self.df".format(attr))
                raise
