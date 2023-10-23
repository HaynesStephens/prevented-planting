# Imports
import datetime
import os
import pandas as pd
import pickle
import pytz
import numpy as np

# Intel patch to accelerate ML algorithms
from sklearnex import patch_sklearn
patch_sklearn()

# Import specific modules from sklearn
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import ParameterSampler

def getTunedModel_leave1out(baseModel, random_state, df_in, features_in, labels_in):
    """
    Trains a random forest regressor with hyperparameters tuned using a leave-one-out cross-validation approach.

    Args:
        baseModel (sklearn.ensemble.RandomForestRegressor): The base random forest regressor to use.
        random_state (int): The random state to use for reproducibility.
        df_in (pandas.DataFrame): The input dataframe containing the features and labels.
        features_in (numpy.ndarray): The input features.
        labels_in (numpy.ndarray): The input labels.

    Returns:
        pandas.DataFrame: A dataframe containing the hyperparameters and performance metrics for each iteration.
    """
    n_estimators = sp_randInt(10, 801)
    max_samples = sp_randFloat()
    min_samples_leaf = sp_randInt(1, 51)
    max_depth = sp_randInt(10, 41)

    random_grid = {'n_estimators': n_estimators,
                   'min_samples_leaf': min_samples_leaf,
                   'max_samples': max_samples,
                   'max_depth': max_depth}

    list_out = []

    years = np.arange(1996, 2023)
    for year in years:
        print('Year: {0}'.format(year))
        param_grid = list(ParameterSampler(random_grid, n_iter=25, random_state=random_state))
        for i, params in enumerate(param_grid):
            # Create a model with a set of params
            model_i = baseModel(random_state=random_state,
                                criterion=criterion,
                                n_estimators=params['n_estimators'],
                                max_depth=params['max_depth'],
                                max_samples=params['max_samples'],
                                min_samples_leaf=params['min_samples_leaf'])
            model_i.fit(features_in[~df_in.year.isin([year])], labels_in[~df_in.year.isin([year])])
            train_score = metrics.mean_squared_error(labels_in[~df_in.year.isin([year])], model_i.predict(features_in[~df_in.year.isin([year])])
            test_score = metrics.mean_squared_error(labels_in[df_in.year.isin([year])], model_i.predict(features_in[df_in.year.isin([year])])
            print('Param iter.: {0} |'.format(i), 'Train: {0:.2f} |'.format(train_score), 'Test: {0:.2f}'.format(test_score))
            list_out.append(dict(year=year, param_run=i, n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                 max_samples=params['max_samples'], min_samples_leaf=params['min_samples_leaf'],
                                 train_score=train_score, test_score=test_score))
        print('\n')
    df_out = pd.DataFrame(list_out)
    return df_out

def get_file_id():
    """
    Returns a string representing the current date and time in the US/Pacific timezone
    formatted as 'YYYY-MM-DD-HH-MM'.
    """
    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    fileid = '{0}-{1}-{2}-{3}-{4}'.format(str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2),
                                          str(now.hour).zfill(2), str(now.minute).zfill(2))
    return fileid

if __name__ == '__main__':
    random_state = 4

    df = pd.read_csv('/project2/moyer/ag_data/prevented-planting/traindata-soy-excessmoist.csv')
    df['fips'] = df.fips.astype(str).str.zfill(5)

    # Set values above 1.0 to 1.0
    df.loc[df["ppfrac"] > 1.0, "ppfrac"] = 1.0

    df['ppval'] = df.ppfrac

    # Regressor data changes
    df = df[df.ppfrac > 0.0]

    state_exc_100lon = ['HI', 'AK', 'WA', 'OR', 'CA', 'ID', 'NV', 'AZ', 'MT', 'WY', 'UT', 'CO', 'NM', 'AS', 'MP', 'PR', 'DC', 'GU', 'VI']
    df = df[~df.state.isin(state_exc_100lon]

    print(len(df))
    # Filter to counties with at least one occurrence of PP
    df = df[df.fips.isin(df[df.ppfrac > 0.0].fips.unique())]
    print(len(df))

    # Split data into labels & features -- and convert to numpy arrays
    # CUSTOM VARIABLES
    labels = np.array(df['ppfrac'])
    months_incl = np.array([1, 2, 3, 4, 5, 6])
    months_excl = np.array([month for month in np.arange(1, 13) if month not in months_incl])
    weather_vars = ['rain_', 'tempair_', 'watersoil_']
    weather_vars = [var + str(month).zfill(2) for var in weather_vars for month in months_incl]
    cst_vars = [
        'frac_tile_drained',
        'lat', 'lon',
        'drain_class',
        'awc_mean',
        'om_mean',
        'clay_mean',
        'ksat_mean',
    ]
    df_features = df[cst_vars + weather_vars]
    print(df_features.columns)
    feature_list = list(df_features.columns)
    features = np.array(df_features)

    criterion = 'squared_error'
    rf = RandomForestRegressor

    df_out = getTunedModel_leave1out(rf, random_state, df, features, labels)

    filename = 'RFregr-' + get_file_id()
    print(filename)
    savedir = '/project2/moyer/ag_data/prevented-planting/Models/RFregr/' + filename

    # Create a new directory for the model run and save blurb.
    os.mkdir(savedir)
    df_out.to_csv(savedir + '/params.csv', index=False)
