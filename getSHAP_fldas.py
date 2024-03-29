# imports
# Here we go
import sys
decade_start = int(sys.argv[1])

import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklego.meta import ZeroInflatedRegressor
import shap

ZIRpart=True

if ZIRpart:
    #intel patch to accelerate ml algorithms
    from sklearnex import patch_sklearn
    patch_sklearn()
    class ZIRpart:
        def __init__(self, rfclass, rfregr):
            def load_model(modeltype, filename):
                with open('/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/model.pkl'.format(modeltype, filename),'rb') as f:
                    rf = pickle.load(f)
                return rf
            self.classifier_ = load_model('RFclass', rfclass)
            self.regressor_ = load_model('RFregr', rfregr)

        def getOutput(self, feature_list, crop='corn', varbase=''):
            output = pd.read_csv('/project2/moyer/ag_data/prevented-planting/traindata-{0}-excessmoist.csv'.format(crop))
            output['fips'] = output.fips.astype(str).str.zfill(5)
            output.loc[output["ppfrac"] > 1.0, "ppfrac"] = 1.0
            state_exc_100lon = ['HI','AK','WA','OR','CA','ID','NV','AZ','MT','WY','UT','CO','NM','AS', 'MP', 'PR', 'DC', 'GU','VI']
            output = output[~output.state.isin(state_exc_100lon)]

            # Create anoms
            if varbase != '':
                varcols = [col for col in output.columns if col.startswith(varbase+'_')]
                varmean = output.groupby('fips')[varcols].mean().reset_index().rename(columns=dict(zip(varcols, [varbase+'mean_'+f.split('_')[1] for f in varcols])))
                output = output.merge(varmean, on='fips')
                for i in range(1,13):
                    month = str(i).zfill(2)
                    print(month, varbase+'anom_'+month)
                    output[varbase+'anom_'+month] = (output[varbase+'_'+month] - output[varbase+'mean_'+month]) / output[varbase+'mean_'+month]

            output['pred_cl'] = self.classifier_.predict(output[feature_list].values)
            output['pred_re'] = self.regressor_.predict(output[feature_list].values)
            output['pred'] = output.pred_cl * output.pred_re
            output['pred_tot'] = output.pred * output.Total
            return output
    crop = 'corn'
    modeltype = 'ZIRpart'
    filename = "ZIRpart_corn_ppreq"
    feature_list = ['frac_tile_drained', 'lat', 'lon', 
                    'drain_class', 'awc_mean', 'om_mean', 'clay_mean', 'ksat_mean',
                    'rain_01', 'rain_02', 'rain_03', 'rain_04', 'rain_05', 'rain_06', 
                    'tempair_01', 'tempair_02', 'tempair_03', 'tempair_04', 'tempair_05', 'tempair_06', 
                    # 'tempairmean_01', 'tempairmean_02', 'tempairmean_03', 'tempairmean_04', 'tempairmean_05', 'tempairmean_06', 
                    # 'tempairanom_01', 'tempairanom_02', 'tempairanom_03', 'tempairanom_04', 'tempairanom_05', 'tempairanom_06', 
                    'watersoil_01', 'watersoil_02','watersoil_03', 'watersoil_04', 'watersoil_05', 'watersoil_06']
    print('Loading ZIRpart model.')
    model = ZIRpart('RFclass-2023-10-14-11-01', 'RFregr-2023-10-14-11-27')
    modeldir = '/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/'.format(modeltype, filename)
else: 
    def load_model(modeltype, filename):
        with open('/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/model.pkl'.format(modeltype, filename),'rb') as f:
            rf = pickle.load(f)
        return rf
    modeltype = 'ZIR'
    filename = "ZIR-2023-08-25-16-15-12"
    model = load_model(modeltype, filename)
    modeldir = '/project2/moyer/ag_data/prevented-planting/Models/{0}/{1}/'.format(modeltype, filename)
    feature_list = pickle.load(open( modeldir+'feature_list.pkl', 'rb' ))

print('Loading output.')
# Historical
output = pd.read_csv(modeldir+'predictions-fldas.csv')
output['fips'] = output.fips.astype(str).str.zfill(5)
output['pred_tot'] = output.pred * output.Total
features_df = output[feature_list]

def saveShapleys(model_in, feature_list, input_data, bkg_data=None):
    print('Model type: ', type(model_in))
    print('Loading TreeExplainer.')
    features_df = input_data[feature_list].copy()
    X = features_df
    if bkg_data is None:
        print('SHAP values from the conditional distribution.')
        explainer = shap.TreeExplainer(model_in, feature_perturbation = "tree_path_dependent")
    else:
        print('SHAP values from the marginal distribution.')
        # Seems to work whether bkg_data is a df or a numpy array, but the df is quicker (~13%).
        explainer = shap.TreeExplainer(model_in, data = bkg_data[feature_list], feature_perturbation = "interventional")

    print('Getting shap values.')
    if str(model_in).split('(')[0] == 'RandomForestClassifier':
        shap_values = explainer(X, check_additivity=False)[:,:,1]
    elif str(model_in).split('(')[0] == 'RandomForestRegressor':
        # Helper procedure
        if hasattr(explainer, "expected_value"):
            if type(explainer.expected_value) is np.ndarray:
                explainer.expected_value = explainer.expected_value.mean()
        shap_values = explainer(X, check_additivity=False)
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

print('Decade: {0}-{1}'.format(decade_start,decade_start+9))
decade_range = np.arange(decade_start,decade_start+10)
input_data = output[output.year.isin(decade_range)]
input_data = input_data.reset_index(drop=True)

bkg_data_class = output.copy()
shap_class = saveShapleys(model.classifier_, feature_list, input_data, bkg_data=bkg_data_class)
shap_class.to_csv(modeldir+'shap_fldas_class_{0}-{1}_bkg.csv'.format(decade_start,input_data.year.max()),index=False)

bkg_data_regr = output[output.pred>0].copy()
shap_regr = saveShapleys(model.regressor_, feature_list, input_data, bkg_data=bkg_data_regr)
shap_regr.to_csv(modeldir+'shap_fldas_regr_{0}-{1}_bkg.csv'.format(decade_start,input_data.year.max()),index=False)