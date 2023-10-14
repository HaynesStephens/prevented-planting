# imports
import pandas as pd
import numpy as np
import datetime
import pytz
import pickle
import os
#intel patch to accelerate ml algorithms
from sklearnex import patch_sklearn
patch_sklearn()
#
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklego.meta import ZeroInflatedRegressor
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import ParameterSampler

def getTunedModel_leave1out( baseModel, random_state, df_in, features_in, labels_in ):
    n_estimators = sp_randInt(10, 801)
    max_samples = sp_randFloat()
    min_samples_leaf = sp_randInt(1, 51)
    max_depth = sp_randInt(10, 41)

    random_grid = {'n_estimators': n_estimators,
                   'min_samples_leaf': min_samples_leaf,
                   'max_samples': max_samples,
                   'max_depth':max_depth}
    #print(random_grid)

    list_out = []

    years = np.arange(1996,2023)
    for year in years:
        print('Year: {0}'.format(year))
        param_grid = list(ParameterSampler(random_grid, n_iter=25, random_state=random_state))
        for i, params in enumerate(param_grid):
            # Create a model with set of params
            model_i = baseModel(random_state = random_state,
                                criterion=criterion,
                                n_estimators = params['n_estimators'],
                                max_depth=params['max_depth'],
                                max_samples = params['max_samples'],
                                min_samples_leaf = params['min_samples_leaf'])
            model_i.fit(features_in[~df_in.year.isin([year])], labels_in[~df_in.year.isin([year])])
            train_score = metrics.accuracy_score(labels_in[~df_in.year.isin([year])], model_i.predict(features_in[~df_in.year.isin([year])]))
            test_score = metrics.accuracy_score(labels_in[df_in.year.isin([year])], model_i.predict(features_in[df_in.year.isin([year])]))
            print('Param iter.: {0} |'.format(i), 'Train: {0:.2f} |'.format(train_score), 'Test: {0:.2f}'.format(test_score))
            list_out.append(dict(year = year, param_run = i, n_estimators = params['n_estimators'], max_depth=params['max_depth'],
                                 max_samples = params['max_samples'], min_samples_leaf = params['min_samples_leaf'],
                                 train_score = train_score, test_score = test_score))
        print('\n')
    df_out = pd.DataFrame(list_out)
    return df_out

# Create unique filename for model run.
def get_file_id():
    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    fileid = '{0}-{1}-{2}-{3}-{4}'.format(str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2),
                                          str(now.hour).zfill(2), str(now.minute).zfill(2))
    return fileid

if __name__=='__main__':

    random_state = 4

    df = pd.read_csv('/project2/moyer/ag_data/prevented-planting/traindata-corn-excessmoist.csv')
    df['fips'] = df.fips.astype(str).str.zfill(5)

    # Set values above 1.0 to 1.0
    df.loc[df["ppfrac"] > 1.0, "ppfrac"] = 1.0
    # # Trim values above 1.0
    # df = df[df.ppfrac<=1.0]

    df['ppval'] = df.ppfrac

    # Classifier data changes
    df.loc[df["ppfrac"] > 0.0, "ppfrac"] = 1.0

    state_exc_100lon = ['HI','AK','WA','OR','CA','ID','NV','AZ','MT','WY','UT','CO','NM','AS', 'MP', 'PR', 'DC', 'GU','VI']
    df = df[~df.state.isin(state_exc_100lon)]

    print(len(df))
    df = df[df.fips.isin(df[df.ppfrac>0.0].fips.unique())]
    print(len(df))


    # Split data into labels & features -- and convert to numpy arrays
    # CUSTOM VARIABLES
    labels = np.array(df['ppfrac'])
    months_incl = np.array([1,2,3,4,5,6])
    months_excl = np.array([month for month in np.arange(1,13) if month not in months_incl])
    weather_vars = ['rain_','tempair_','watersoil_'] #['evaptrans_','runsurf_','runsub_','rain_','tempair_','watersoil_','tempsoil_']
    weather_vars = [var+str(month).zfill(2) for var in weather_vars for month in months_incl]
    cst_vars = [
        'frac_tile_drained',
        'lat', 'lon', #'fips',
        # 'aquifer_glacial', 'aquifer_uncon', 'aquifer_semicon',
        # 'aquifer_glacial_pct', 'aquifer_uncon_pct', 'aquifer_semicon_pct',
        'drain_class',
        'awc_mean',#'awc_mean_0_5', 'awc_mean_5_15', 'awc_mean_15_30', 'awc_mean_30_60', 'awc_mean_60_100',
        'om_mean',#'om_mean_0_5', 'om_mean_5_15', 'om_mean_15_30', 'om_mean_30_60', 'om_mean_60_100',
        'clay_mean',#'clay_mean_0_5', 'clay_mean_5_15', 'clay_mean_15_30', 'clay_mean_30_60', 'clay_mean_60_100',
        'ksat_mean',#'ksat_mean_0_5', 'ksat_mean_5_15', 'ksat_mean_15_30', 'ksat_mean_30_60', 'ksat_mean_60_100',
        ]
    df_features = df[cst_vars+weather_vars]
    print(df_features.columns)
    feature_list=list(df_features.columns)
    features=np.array(df_features)


    criterion='gini'
    rf = RandomForestClassifier

    df_out = getTunedModel_leave1out(rf, random_state, df, features, labels)

    filename = 'RFclass-' + get_file_id()
    print(filename)
    savedir = '/project2/moyer/ag_data/prevented-planting/Models/RFclass/'+filename

    # Create new directory for model run and save blurb.
    os.mkdir(savedir)
    df_out.to_csv(savedir+'/params.csv', index=False)
