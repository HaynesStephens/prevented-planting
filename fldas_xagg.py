import numpy as np
import pandas as pd
import xagg as xa
import geopandas as gpd
import xarray as xr


def getFLDAS():
    dirPath = '/project2/moyer/ag_data/fldas'
    fileName = 'FLDAS_NOAH01_C_GL_M.*.nc'
    fullfilename = dirPath+fileName
    print(fullfilename)
    return xr.open_mfdataset(fullfilename)

def getGDF():
    gdf = gpd.read_file( '/project2/moyer/ag_data/cb_2021_us_county_500k/' )
    return gdf


def reformat(aggregated_subset, varname, start_date='1982-01-01'):
    df          = aggregated_subset.to_dataframe().melt(id_vars=['disID' , 'state' , 'district' ,'statename'])
    df['date']  = pd.to_datetime(start_date) + pd.to_timedelta(df.variable.str.lstrip(varname).astype(int), unit='D')
    df          = df.drop('variable', 1).rename(columns={'value':varname})
    df['Year']  = df.date.dt.year
    df['doy']   = df.date.dt.dayofyear
    return df


if __name__=='__main__':

    print('LOAD DISTRICTS')
    geodf = getGDF()
    print('\n')

    print('OPEN FLDAS')
    da = getFLDAS()#.sel(time=slice('1982', '2022'))
    print('opened.')

    # print('CREATE WEIGHTMAP')
    # weightmap = xa.pixel_overlaps( da, geodf, weights=maizearea )
    # print('AGGREGATE DATA')
    # aggregated = xa.aggregate( da, weightmap )
    # print('CREATE DATAFRAME')
    # df = reformat(aggregated, varname)
    # print('SAVE DF')
    # df = df.drop_duplicates()
    # # df.to_csv('/project2/moyer/ag_data/growing-seasons/ggcmi-regrid/{0}_T6.csv'.format(varname), index=False)
    # print('COMPLETE.\n')


