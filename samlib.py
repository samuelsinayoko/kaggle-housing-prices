"""An ML library to help with Kaggle problems.
"""
from itertools import chain
import logging

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import sklearn.base



logger = logging.getLogger(__name__)


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
        logger.warn('DeprecationWarning: use `pd.concat([raw_train, raw_test], keys=["train", "test"])` '
                    'instead of this class')
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



def fillna(df, wok, columns=None, **kwargs):
    """Apply callable parameter `wok` to fill the null values in `df`.

    Can be used in a pipe chain. This is unlike the default pandas
    dataframe `fillna` method that does not take a callable as a
    parameter nor a `columns` parameter. The `columns` argument allows
    one to apply the callable to only a subset of the input dataframe
    (while still returning a dataframe of the same shape as the
    original dataframe).

    Parameters
    ----------
    df: DataFrame (or Series)
         The object we want to fill the null values in.
    wok: callable or string
        Aggregate function, of the form df -> df' = foo(df), used to
        fill the null values. If it's a string, apply the method whose
        name is wok. Examples: np.nanmean, np.nanmedian, 'mean',
        'median'.
    columns: List[str], optional
        The list of columns to fill the null values in.
    kwargs: dict, optional
        Optional arguments passed to pd.fillna.

    Returns
    -------
    A dataframe having the same shape as df.

    """
    if columns is not None:
        df = df.copy()
        df.loc[:, columns] = fillna(df.loc[:, columns], wok, **kwargs)
        return df

    if callable(wok):
        return df.fillna(wok(df), **kwargs)
    elif isinstance(wok, str):
        if hasattr(df, wok):
            return df.fillna(getattr(df, wok)(), **kwargs)
        else:
            raise TypeError('{!r} method not found in {!r}', wok, df)


def has_nulls(df):
    """Return boolean indicating whether input dataframe `df` has got null values"""
    return df.isnull().sum().any()

# Pipeline
# Create a pipeline so we can process all the data later in one go if needed
class Pipeline:
    """Add function to transform input dataframe. Functions can be appended to the pipeline,
    and the pipeline can be called to apply all the functions successively.

    When adding a functino to the pipeline, the function is only added
    if it is not at the end of the pipeline already. That way we can
    re-run a cell in a notebook multiple times (makes the pipeline a
    little more indempotent).

    """
    def __init__(self, df):
        self.input_df = df
        self._pipeline = []

    def append(self, func):
        """Append function to pipe"""
        if not self._pipeline or not self._same_func(self._pipeline[-1], func):
            self._pipeline.append(func)

    def __call__(self):
        """Run the pipe"""
        df = self.input_df
        for func in self._pipeline:
            df = df.pipe(func)
        return df

    @staticmethod
    def _same_func(f1, f2):
        return f1.__name__ == f2.__name__

    def __str__(self):
        return str(self._pipeline)

    def __repr__(self):
        return repr(self._pipeline)

    def __eq__(self, dg):
        """Compare the `dg` dataframe with the pipeline output"""
        return all(self() == dg)



# Plotting functions
def violinplot(df, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.violinplot(df, ax=ax)
    for xlab in ax.get_xticklabels():
        xlab.set_rotation(30)



## Visualisation
def featureplot(df, nrows=1, ncols=1, figsize=(12,8), plotfunc=sns.violinplot, **kwargs):
    """Plot the dataframe features.
    Use Matplotlib to plot individual features accross columns.
    """
    width, height = figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * nrows));
    i = 0
    plots_per_figure = max(df.shape[1] // (nrows * ncols), 1)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    if nrows > 1 and ncols > 1:
        axes = chain.from_iterable(axes)  # flatten the nested list
    for j, ax in zip(range(plots_per_figure, df.shape[1] + 1, plots_per_figure), axes):
        plotfunc(data=df.iloc[:, i:j], ax=ax, **kwargs)
        i = j
    plt.tight_layout()

def featureplots(df, nrows=1, ncols=1, figsize=(12,8), plotfuncs=(sns.violinplot,), axis=1, **kwargs):
    """Plot the dataframe features.
    Use Matplotlib to plot individual features accross columns.
    """
    width, height = figsize
    nfuncs = len(plotfuncs)

    if axis == 1:
        # Cycle plotfuncs accross columns
        a = nfuncs
        b = 1
        funclst = plotfuncs * nrows * ncols
    elif axis == 0:
        # Cycle plotfuncs accross rows
        a = 1
        b = nfuncs
        funclst = zip(*(plotfuncs * ncols))
    else:
        ValueError('axis must be 0 or 1')
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols * a, height * nrows * b));
    i = 0
    plots_per_figure = max(df.shape[1] // (nrows * ncols), 1)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    if nrows > 1 and ncols > 1:
        axes = chain.from_iterable(axes)  # flatten the nested list
    # TODO: must generalize this so we can loop over rows. Possibly use a gridspec instead of fig,axes?
    for j, ax in zip(range(plots_per_figure, df.shape[1] + 1, plots_per_figure), axes):
        plotfunc(data=df.iloc[:, i:j], ax=ax, **kwargs)
        i = j
    plt.tight_layout()

## - using Seaborn and long form data (5 times slower than featureplot)
def featureplot2(df, ncols=1, size=5, aspect=0.5, plotfunc=sns.violinplot,
                 hook=None, **map_kwargs):
    """Plot the dataframe features.
    Transform the input dataframe to long form and use
    Seaborn's FacetGrid to plot individual features accross columns.
    """
    df = df.copy()
    if df.columns.name is None:
        feature = df.columns.name = 'col'
    else:
        feature = df.columns.name

    # Transform input dataframe to long form data
    lf = df.stack().reset_index(name="value")
    lf = lf.drop([c for c in lf.columns if c.startswith('level_')], axis=1)
    # Visualize with Seaborn
    g = sns.FacetGrid(lf, col=feature, hue=feature,
                      sharex=False, sharey=False, col_wrap=ncols,
                      size=size, aspect=aspect)
    h = g.map(plotfunc, "value", **map_kwargs).set_titles("{col_name}")
    if hook is not None:
        h = hook(h)
    return h
