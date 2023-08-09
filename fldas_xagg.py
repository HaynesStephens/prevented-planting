import numpy as np
import pandas as pd
import xagg as xa
import geopandas as gpd
import xarray as xr


def getFLDAS(year):
    dirPath = '/project2/moyer/ag_data/fldas/'
    fileName = 'FLDAS_NOAH01_C_GL_M.A{0}*.nc'.format(year)
    fullfilename = dirPath+fileName
    print(fullfilename)
    oldnames = ['Evap_tavg', 'Qs_tavg', 'Qsb_tavg', 'Rainf_f_tavg', 'Tair_f_tavg', 'SoilMoi00_10cm_tavg', 'SoilTemp00_10cm_tavg']
    newnames = ['evaptrans', 'runsurf', 'runsub', 'rain', 'tempair', 'watersoil', 'tempsoil']
    namedict = dict(zip(oldnames+['X','Y'], newnames+['lon','lat']))
    ds = xr.open_mfdataset(fullfilename)
    ds = ds[oldnames].rename(namedict)
    return ds

def getGDF():
    gdf = gpd.read_file( '/project2/moyer/ag_data/cb_2021_us_county_500k/' )
    state_exc_100lon = ['HI','AK','WA','OR','CA','ID','NV','AZ','MT','WY','UT','CO','NM','AS', 'MP', 'PR', 'DC', 'GU','VI']
    gdf = gdf[~gdf.STUSPS.isin(state_exc_100lon)]
    gdf = gdf[['STUSPS','GEOID','geometry']]
    return gdf


def reformat(aggregated_subset, year):
    df = aggregated_subset.to_dataframe()
    df['year'] = year
    names = ['evaptrans', 'runsurf', 'runsub', 'rain', 'tempair', 'watersoil', 'tempsoil']
    for name in names:
        oldcols = ['{0}{1}'.format(name,str(i)) for i in range(12)] # Original column names output
        newcols = ['{0}_{1}'.format(name,str(i+1).zfill(2)) for i in range(12)] # New month column names
        df = df.rename(columns=dict(zip(oldcols, newcols)))
    df = df.rename(columns={'GEOID':'fips'})
    return df


if __name__=='__main__':

    print('LOAD DISTRICTS')
    geodf = getGDF()
    print('\n')

    print('OPEN FLDAS')
    for year in np.arange(1982,2023):
        print(year, 'OPENING')
        ds = getFLDAS(year).sel( lat=slice(24,50), lon=slice(-107,-66) )
        print('opened.')

        print('CREATE WEIGHTMAP')
        weightmap = xa.pixel_overlaps( ds, geodf )
        print('AGGREGATE DATA')
        aggregated = xa.aggregate( ds, weightmap )
        df = aggregated.to_dataframe().to_csv('/project2/moyer/ag_data/fldas/_test.csv'.format(year))
        print('CREATE DATAFRAME')
        df = reformat(aggregated, year)
        print('SAVE DF')
        df = df.drop_duplicates()
        df.to_csv('/project2/moyer/ag_data/fldas/fldas_{0}.csv'.format(year), index=False)
        print('COMPLETE.\n')


