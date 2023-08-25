# UPDATES: 
# 1. Constants and watersoil

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
zir = ZeroInflatedRegressor(
    classifier=RandomForestClassifier(random_state=random_state, criterion=classifier_criterion),
    regressor=RandomForestRegressor(random_state=random_state, criterion=regressor_criterion)
)


def getTunedModel( baseModel, random_state ):
    random_grid = {
        'regressor__n_estimators': sp_randInt(10, 801),
        'regressor__min_samples_leaf': sp_randInt(1, 51),
        'regressor__max_depth': sp_randInt(10, 41),
        'regressor__max_samples': sp_randFloat(),
        'classifier__n_estimators': sp_randInt(10, 801),
        'classifier__min_samples_leaf': sp_randInt(1, 51),
        'classifier__max_depth': sp_randInt(10, 41),
        'classifier__max_samples': sp_randFloat()
        }
    print(random_grid)
    model_tuned = RandomizedSearchCV(cv=5, estimator = baseModel, param_distributions = random_grid, n_iter = 25, verbose=1, random_state=random_state , n_jobs = -1)
    return model_tuned


zir_tuned = getTunedModel( zir, random_state )
# Run tuning to find optimal hyperparameters
zir_tuned.fit(train_features,train_labels)


## Select best parameters and fit model
result = pd.DataFrame.from_dict(zir_tuned.cv_results_)
# Choose the best hyperparameters from the random CV search.
best = result[result.rank_test_score == 1]
key = list(best.params.keys())[0]
best_params = dict(best.params)[key]
best_params


# Create a new model instance with the optimal hyperparameters.
zir_opt = ZeroInflatedRegressor(
    classifier=RandomForestClassifier(
        random_state=random_state, criterion=classifier_criterion,
        n_estimators = best_params['classifier__n_estimators'],
        max_samples = best_params['classifier__max_samples'],
        min_samples_leaf = best_params['classifier__min_samples_leaf'],
        max_depth = best_params['classifier__max_depth'],
        ),
    regressor=RandomForestRegressor(
        random_state=random_state, criterion=regressor_criterion, 
        n_estimators = best_params['regressor__n_estimators'], 
        max_samples = best_params['regressor__max_samples'], 
        min_samples_leaf = best_params['regressor__min_samples_leaf'],
        max_depth = best_params['regressor__max_depth']
        )
)

zir_opt.fit(features, labels)
y_pred = zir_opt.predict(features)


# Add performance metrics to the blurb output.
blurb = 'ZIR model (split train-test): 25-iter CV.'
blurb = blurb + '\nConstants and watersoil.'
blurb = blurb + '\nRandom State: {0}'.format(random_state)
blurb = blurb + '\nGoodness of Fit (R2): {0}'.format(metrics.r2_score(labels, y_pred))
blurb = blurb + '\nMean Absolute Error (MAE): {0}'.format(metrics.mean_absolute_error(labels, y_pred))
blurb = blurb + '\nMean Squared Error (MSE): {0}'.format(metrics.mean_squared_error(labels, y_pred))
blurb = blurb + '\nRoot Mean Squared Error (RMSE): {0}'.format(np.sqrt(metrics.mean_squared_error(labels, y_pred)))
mape = np.mean(np.abs((labels - y_pred) / np.abs(labels+0.001)))
blurb = blurb + '\nMean Absolute Percentage Error (MAPE): {0}'.format(round(mape * 100, 2))
blurb = blurb + '\nAccuracy: {0}'.format(round(100*(1 - mape), 2))
print(blurb)


# Create unique filename for model run.
def get_file_id():
    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    fileid = '{0}-{1}-{2}-{3}-{4}-{5}'.format(str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2), 
                                          str(now.hour).zfill(2), str(now.minute).zfill(2), str(now.second).zfill(2))
    return fileid
filename = 'ZIR-' + get_file_id()
savedir = '/project2/moyer/ag_data/prevented-planting/Models/ZIR/'+filename

# Create new directory for model run and save blurb.
os.mkdir(savedir)
with open(savedir+'/run_notes.txt', 'w') as f:
    f.write(blurb)
    f.write('\n\n')
    for item in feature_list:
        f.write("{0}\n".format(item))
with open(savedir+'/feature_list.pkl', 'wb') as f:
    pickle.dump(feature_list, f)
# Save fitted model.
filepath = savedir+'/model.pkl'
pickle.dump(zir_opt, open(filepath, 'wb'))
# Predict historical data and add to dataset.
df['pred'] = zir_opt.predict(np.array(df[feature_list]))
df['pred_cl'] = zir_opt.classifier_.predict(np.array(df[feature_list])).astype(int)
df['pred_re'] = zir_opt.regressor_.predict(np.array(df[feature_list]))


# Save new dataframe with saved model.
df.to_csv(savedir+'/predictions-fldas.csv',index=False)
