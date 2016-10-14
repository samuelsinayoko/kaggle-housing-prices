# House prices advanced regression techniques
Exploring the Kaggle house prices competition
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Virtual environment

Use conda to create the virtual environment

```
conda env create --file=environment.yml
```

To update the environment

```
conda env update --file=environment.yml
```

## nbstripout
Use nbstripout (https://pypi.python.org/pypi/nbstripout) to strip the output from the notebooks before committing (makes logs more readable). Run the following in your repository to activate (`nbstripout` should have been pip installed in the virtual environment)

```
nbstripout --install
```

# TODO

- Refactor data exploration notebook
 + Move regression to another notebook
 + Save transformed data to csv file
 + Save useful functions/classes into a library
- Explore categorical features and identify key ones
- Encode categorical features and include them into a more complex regression model
