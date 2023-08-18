# imports
import numpy as np
import pandas as pd
import pickle
import os
import time
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklego.meta import ZeroInflatedRegressor
import xgboost as xgb
import os, json, fnmatch, xarray as xr, numpy as np, plotly.express as px, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl
from urllib.request import urlopen
import shap

def load_model(modeltype, filename):
  with open('/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/model.pkl'.format(modeltype, filename),'rb') as f:
    rf = pickle.load(f)
    return rf

modeltype = 'ZIR'
filename = "ZIR-2023-08-15-15-39-06"
figdir = '/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/'.format(modeltype, filename)
model = load_model(modeltype, filename)

feature_list = pickle.load(open(
    '/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/feature_list.pkl'.format(modeltype, filename),
    'rb'
))

# Historical
output = pd.read_csv(
    '/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/predictions-fldas.csv'.format(modeltype, filename)
)
output['fips'] = output.fips.astype(str).str.zfill(5)
output['pred_tot'] = output.pred * output.Total

output = output[:30]

features_df = output[feature_list]

def saveShapleys(input_data, feature_list, model_in):
    print('Model type: ', type(model_in))
    print('Loading Explainer.')
    features_df = input_data[feature_list].copy()
    X = features_df
    explainer = shap.Explainer(model_in)

    print('Getting shap values.')
    if type(model_in) == sklearn.ensemble._forest.RandomForestClassifier:
        shap_values = explainer(X)[:,:,1]
    elif type(model_in) == sklearn.ensemble._forest.RandomForestRegressor:
        # Helper procedure
        if hasattr(explainer, "expected_value"):
            if type(explainer.expected_value) is np.ndarray:
                explainer.expected_value = explainer.expected_value.mean()
        shap_values = explainer(X)
    else:
        raise( AssertionError("ERROR: Wrong model type used."))
    
    print('Composing output df.')
    feature_names = X.columns
    dfshap = pd.DataFrame(shap_values.values, columns = [f+'_SHAP' for f in feature_names])
    out = pd.DataFrame(shap_values.base_values, columns = ['base_SHAP'])
    out = pd.concat([input_data[['fips','year']].copy(), out, dfshap], axis=1)
    out['pred_SHAP'] = out[[f for f in out.columns if '_SHAP' in f]].sum(axis=1)
    print('Complete.')
    return out

shap_class = saveShapleys(output, feature_list, model.classifier_)
shap_class.to_csv('shap_class.csv',index=False)

shap_regr = saveShapleys(output, feature_list, model.regressor_)
shap_regr.to_csv('shap_regr.csv',index=False)