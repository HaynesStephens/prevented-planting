# imports
import pandas as pd
import numpy as np
import datetime
import pytz
import pickle
import os
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklego.meta import ZeroInflatedRegressor
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

random_state = 4

df = pd.read_csv('/project2/moyer/ag_data/prevented-planting/traindata-corn-excessmoist.csv')
df['fips'] = df.fips.astype(str).str.zfill(5)
# Set values above 1.0 to 1.0
df.loc[df["ppfrac"] > 1.0, "ppfrac"] = 1.0
state_exc_100lon = ['HI','AK','WA','OR','CA','ID','NV','AZ','MT','WY','UT','CO','NM','AS', 'MP', 'PR', 'DC', 'GU','VI']
df = df[~df.state.isin(state_exc_100lon)]


# Split data into labels & features -- and convert to numpy arrays
# CUSTOM VARIABLES
labels = np.array(df['ppfrac'])
months_incl = np.array([1,2,3,4,5,6])
months_excl = np.array([month for month in np.arange(1,13) if month not in months_incl])
weather_vars = ['watersoil_'] #['evaptrans_','runsurf_','runsub_','rain_','tempair_','watersoil_','tempsoil_']
weather_vars = [var+str(month).zfill(2) for var in weather_vars for month in months_incl]
cst_vars = [
    'frac_tile_drained', #'lat', 'lon', 'fips',
    'drain_class',
    'awc_mean',#'awc_mean_0_5', 'awc_mean_5_15', 'awc_mean_15_30', 'awc_mean_30_60', 'awc_mean_60_100', 
    'om_mean',#'om_mean_0_5', 'om_mean_5_15', 'om_mean_15_30', 'om_mean_30_60', 'om_mean_60_100',
    'clay_mean',#'clay_mean_0_5', 'clay_mean_5_15', 'clay_mean_15_30', 'clay_mean_30_60', 'clay_mean_60_100', 
    #'ksat_mean',#'ksat_mean_0_5', 'ksat_mean_5_15', 'ksat_mean_15_30', 'ksat_mean_30_60', 'ksat_mean_60_100',
    ]
df_features = df[cst_vars+weather_vars]
print(df_features.columns)
feature_list=list(df_features.columns)
features=np.array(df_features)


# RANDOM SPLIT
train_features, test_features, train_labels, test_labels = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.20, 
                                                                            random_state = random_state, 
                                                                            shuffle = True)


# Create a reference model to be tuned.
classifier_criterion='gini'
regressor_criterion='squared_error'