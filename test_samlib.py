import pandas as pd
import samlib
import pytest

@pytest.fixture
def raw_train():
    return pd.read_csv('data/train_prepared_light.csv')

@pytest.fixture
def raw_test():
    return pd.read_csv('data/test_prepared_light.csv')

@pytest.fixture
def ds(raw_train, raw_test):
    return samlib.DataSet(raw_train, raw_test)



def test_split_merge(ds, raw_train, raw_test):
    """Check the merge and split functions"""
    df1, df2 = ds.split(ds.df)
    assert all(df1 == raw_train)
    assert all(df2 == raw_test)
    assert all(ds.merge(df1, df2) == ds.df)


def test_synchronization(ds):
    """Check that if we update df then the train and test sets are
    updated accordingly
    """
    ds.df = 2 * ds.df
    assert all(ds.train == 2 * ds.raw_train)
    assert all(ds.test == 2 * ds.raw_test)


def test_copy(ds):
    ds1 = ds
    ds2 = ds1.copy()
    assert not (ds1 is ds2)
    assert all(ds1.df == ds2.df)
    assert all(ds1.raw_train == ds2.raw_train)
    assert all(ds1.train == ds2.train)
    assert all(ds1.test == ds2.test)


def test_apply(ds):
    ds2 = ds.apply(lambda x: x * 2)
    assert not (ds is ds2)
    assert all(ds.df == ds2.df * 2)


def test_apply_inplace(ds):
    ds_init = ds.copy()
    ds2 = ds.apply(lambda x: x * 2, inplace=True)
    assert (ds is ds2)
    assert all(ds2.df == ds_init.df * 2)
    assert all(ds2.raw_train == ds_init.raw_train)


def test_getattr(ds):
    """Get an attribute of the underlying dataframe if possible"""
    assert all(ds.columns == ds.df.columns)
    assert ds.shape == ds.df.shape
