import pandas as pd
import numpy as numpy
import tqdm
from IPython import embed
import matplotlib.pyplot as plt
import time



def integrate(df_feature, df_durations, rowScan, integrationType ='abs'):
    """
    Integrate a variable over a given time period.
    The type specifies if the integration should be over absolute change, or positive and negative.
    
    types: pos,neg,all

    """
    timeLastCorrection  = df_durations[df_durations.date == rowScan.dateLastCorrection].end_observing.values[0]
    timeStartObserving  = df_durations[df_durations.date == rowScan.obs_date   ].start_observing.values[0]

    df_feature = df_feature.loc[(df_feature.date > timeLastCorrection) & (df_feature.date < timeStartObserving)]
    if integrationType == 'pos':
        return df_feature.loc[: , df_feature.columns != 'date'].diff(1).clip(lower=0).sum(min_count = 1, skipna = True).get(0)
            

    elif integrationType == 'neg':
        return df_feature.loc[: , df_feature.columns != 'date'].diff(1).clip(upper=0).sum(min_count = 1, skipna = True).get(0)
        
    
    elif integrationType == 'abs':
        return df_feature.loc[: , df_feature.columns != 'date'].diff(1).abs().sum(min_count = 1, skipna = True).get(0)
        



def tilttemp():
    df_temp = pd.read_csv('./Data/db_exports/TILT1T.csv')
    df_daz = pd.read_csv('./Data/db_exports/DAZ_TILTTEMP.csv')
    df_del = pd.read_csv('./Data/db_exports/DEL_TILTTEMP.csv')

    df_temp['date'] = pd.to_datetime(df_temp['date'])
    df_daz['date'] = pd.to_datetime(df_daz['date'])
    df_del['date'] = pd.to_datetime(df_del['date'])

    t1 = pd.to_datetime('2022-01-26 12:41:18')
    t2 = pd.to_datetime('2022-01-26 23:51:55')

    df_temp1 = df_temp.loc[(df_temp.date > t1) & (df_temp.date < t2)]
    df_daz1 = df_daz.loc[(df_daz.date > t1) & (df_daz.date < t2)]
    df_del1 = df_del.loc[(df_del.date > t1) & (df_del.date < t2)]

    #plot all 3 dfs in same plot
    fig, ax = plt.subplots(3,1, figsize=(10,10))
    ax[0].plot(df_temp1.date, df_temp1.TILT1T, label='TILT1T')
    ax[1].plot(df_daz1.date, df_daz1.DAZ_TILTTEMP, label='DAZ_TILTTEMP')
    ax[2].plot(df_del1.date, df_del1.DEL_TILTTEMP, label='DEL_TILTTEMP')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig('tmptest.png')

def azel():
    df_az = pd.read_csv('./Data/db_exports/ACTUALAZ.csv')
    df_el = pd.read_csv('./Data/db_exports/ACTUALEL.csv')

    df_az['date'] = pd.to_datetime(df_az['date'])
    df_el['date'] = pd.to_datetime(df_el['date'])

    t1 = pd.to_datetime('2022-01-26 12:41:18')
    t2 = pd.to_datetime('2022-01-26 23:51:55')

    df_az1 = df_az.loc[(df_az.date > t1) & (df_az.date < t2)]
    df_el1 = df_el.loc[(df_el.date > t1) & (df_el.date < t2)]

    #plot all 3 dfs in same plot
    fig, ax = plt.subplots(2,1, figsize=(10,10))
    ax[0].plot(df_az1.date, df_az1.ACTUALAZ, label='AZ')
    ax[1].plot(df_el1.date, df_el1.ACTUALEL, label='EL')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('azeltest.png')


def get_integration_features(features, integrationType, df_scans):

    df_features = pd.DataFrame({'date': df_scans['obs_date']})

    df_durations = pd.read_csv('./Data/df_scanDuration.csv')
    df_durations['date'] = pd.to_datetime(df_durations['date'])
    df_durations['start_observing'] = pd.to_datetime(df_durations['start_observing'])
    df_durations['end_observing']   = pd.to_datetime(df_durations['end_observing'])


    for feature in features:

        integratedFeature = []
        df_feature = pd.read_csv(f'./Data/db_exports/{feature}.csv')
        df_feature['date'] = pd.to_datetime(df_feature['date'])


        for rowScan in tqdm.tqdm(df_scans.itertuples()):
            integratedFeature.append(integrate(df_feature, df_durations, rowScan, integrationType))

        df_features[feature] = integratedFeature

    df_features.to_csv(f'./Data/processed_v2/integrated_{integrationType}.csv', index=False)


if __name__ == '__main__':

    features = ['TILT1T', 'DAZ_TILTTEMP', 'DEL_TILTTEMP', 'ACTUALAZ', 'ACTUALEL']

    df_scans = pd.read_csv('./Data/tmp2022_clean.csv')    
    df_scans['obs_date'] = pd.to_datetime(df_scans['obs_date'])
    df_scans['dateLastCorrection'] = pd.to_datetime(df_scans['dateLastCorrection'])

    get_integration_features(features, 'abs', df_scans)
    pass
