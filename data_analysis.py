from concurrent.futures import process
from turtle import color
import functions
import importlib
importlib.reload(functions)
from functions import *
from functions import random_seed
from settings import patches, features, dataset_params
import matplotlib.dates as mdates
from matplotlib import rcParams

plt.rcParams['axes.titlesize'] = 18; plt.rcParams['axes.labelsize'] = 18;
plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18

random.seed(random_seed)
np.random.seed(random_seed)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter('ignore', category=UserWarning)



path_unprocessed_df = './Data/' + 'all_scans_15_3.csv'
path_processed_sun  = './Data/' + 'sun_features_15_3.csv'
path_processed_wind = './Data/' + 'wind_features_15_3.csv'

path_median         = './Data/' + 'scans_median_0_0_all.csv'
path_median_sun     = './Data/' + 'scans_sun_median_0_0_all.csv'
path_median_wind    = './Data/' + 'scans_wind_median_0_0_all.csv'
path_merged         = './Data/' + 'merged_features_all.csv'


# if os.environ['COMPUTERNAME'] == 'DESKTOP-018R091':
#     path_tiltmeter_dumps = 'C:/Users/bendi/General/master/code/Data/Tiltmeter/'
# else:
#     path_tiltmeter_dumps = 'C:/Users/bendi/PP/master/code/Data/Tiltmeter.tar/Tiltmeter/'

class Analysis():
    """
    Class for analysis of data. Reads given dataframe and performs analysis on it.
    """
    def __init__(self,
                 path_df         = './Data/merged_features_all.csv',
                 patch_key       = 0,
                 instrument      = None,
                 remove_outliers = True
                ) -> None:

        self.PATH_PAIRS       = f'./Results_optical/{patch_key}/Plots/PairPlots/'
        self.PATH_CORRELATION = f'./Results_optical/{patch_key}/Plots/Correlation/'
        self.PATC_SCANS       = f'./Results_optical/{patch_key}/Plots/ScanLocations/'

        self.name       = path_df.split('/')[-1].split('.')[0]
        self.patch      = patches[patch_key]
        self.instrument = instrument

        df_scans = pd.read_csv("./Data/tmp2022_clean.csv") # Data with offsets from scans
        df       = pd.read_csv(path_df)                  # Data with values from instruments

        if instrument is not None:
            df_scans = df_scans.loc[ df_scans['rx'] == instrument ]

        df_scans.rename(columns={'obs_date':'date'}, inplace=True)
        df_scans['date'] = pd.to_datetime(df_scans['date'])
        df      ['date'] = pd.to_datetime(df      ['date'])
        df_scans = df_scans.loc[:,['Off_Az_new', 'Off_El_new', 'ca_new', 'ie_new', 'date', 'Az', 'El'] ]

        print('Length of df with scans', len(df_scans), '| Length of df with features', len(df))
        
        df_merged = df.merge(df_scans, how ='inner', left_index = True, right_index = True)
        df_merged.rename(columns = {'date_x': 'date'}, inplace = True)
        del df_merged['date_y']
        df_merged.drop_duplicates(inplace = True)
        df_merged.insert( len(df_merged.columns), 'Offset', np.sqrt( df_merged['Off_Az_new']**2 + df_merged['Off_El_new']**2 ) )
        df_merged.insert( len(df_merged.columns)-4, 'HOUR'  , df_merged['date'].dt.hour )
        #df_merged.dropna(inplace = True)


        if self.patch is not None:
            self.df_unfiltered = df_merged.copy()
            df_merged = self.filter_patch(df = df_merged, patch = self.patch)   

        df_merged = df_merged.loc[: , ~df_merged.columns.isin(['Az','El'])]

        self.df_merged = df_merged
        
        if remove_outliers:
            self.remove_outliers()
        #Check for nan values in df
        # if df_merged.isnull().values.any():
        #     print("Nan values in df")
        #     print(df_merged.isnull().sum())

        #print('----- Description of merged DataFrame -----')
        #print(df_merged.describe())



    def __repr__(self): 
        return self.name

    def remove_outliers(self):
        outlier_cols = ['Off_El_new', 'Off_Az_new'] # one or more
        df = self.df_merged
        factor = 1.7
        Q1 = df.loc[: , df.columns.isin(outlier_cols)].quantile(0.25)
        Q3 = df.loc[: , df.columns.isin(outlier_cols)].quantile(0.75)
        IQR = Q3 - Q1

        ## Will raise ValueError in the future
        df = df[~((df.loc[: , df.columns.isin(outlier_cols)] < (Q1 - factor * IQR)) |(df.loc[: , df.columns.isin(outlier_cols)] > (Q3 + factor * IQR))).any(axis=1)]

        self.df_merged = df
        return 

    def filter_patch(self, patch: tuple, df = None, rotation = 23):
        """
        Filters self.df to only include data from a patch
        - If len(patch) is 4 -> Filters from left right top bottom with az and el
        - If len(patch) is 2 -> Transform into cartesian coordinates, rotate around 
          x-axis such that the lines are perpendicular to y-axis, then filter between the two y-values.
        
        """

        if len(patch) == 4:
            l,r,t,b = patch
            df = df.loc[ (df['Az'] > l) & (df['Az'] < r) & (df['El'] > b) & (df['El'] < t) ]
            df = df.loc[ : , ~df.columns.isin(['Az', 'El'])]

        elif len(patch) == 2:
            Az = np.deg2rad(df['ACTUALAZ'])
            El = np.deg2rad(df['ACTUALEL'])

            #x = np.sin(Az.values) * np.cos(El.values)
            y = np.cos(Az.values) * np.cos(El.values)
            z = np.sin(El.values)  

            #x = x
            y = y * np.cos(np.deg2rad(rotation)) - z * np.sin(np.deg2rad(rotation))
            #z = y * np.sin(np.deg2rad(rotation)) + z * np.cos(np.deg2rad(rotation))

            df.insert(0, 'y', y)
            df = df.loc[ (df['y'] > patch[0]) & df['y'] < patch[1], df.columns != 'y' ]

        return df
    def calculate_pvalues(self, df, method='spearman'):
        """
        Calculates the pvalue for the correlation between
        the columns in the dataframe.
        """
        action = {
            'pearson': pearsonr,
            'spearman': spearmanr
        }
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')

        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                pvalues[r][c] = round(action[method](tmp[r], tmp[c])[1], 4)
        return pvalues
        
    def plot_correlation_sun(self):
        dfm = self.df_merged
        dfs = [dfm.loc[dfm['SunAboveHorizon'] == 0, dfm.columns != 'SunAboveHorizon'], dfm.loc[dfm['SunAboveHorizon'] == 1, dfm.columns != 'SunAboveHorizon']]

        for i,df in enumerate(dfs):
            print(i,len(df))
            if len(df) == 0:
                continue
            self.plot_correlation(name = self.name + f'_sunAbove_{i}', df = df)
        
        return 

    def plot_pairs_sun(self):
        dfm = self.df_merged
        dfs = [dfm.loc[dfm['SunAboveHorizon'] == 0, dfm.columns != 'SunAboveHorizon'], dfm.loc[dfm['SunAboveHorizon'] == 1, dfm.columns != 'SunAboveHorizon']]

        for i,df in enumerate(dfs):
            if len(df) == 0:
                continue
            self.plot_pairs(name = self.name + f'_sunAbove_{i}', df = df)
        
        return

    def plot_correlation(self, name=None, df = None, targets = ['Offset', 'Off_Az_new', 'Off_El_new'], pval=False):
        if len(self.df_merged) < 100:
            embed()
            print('Not enough data to plot correlation')
            return

        if name is None:
            name = self.name

        if df is None:
            df = self.df_merged

        if not os.path.exists(self.PATH_CORRELATION):
            os.makedirs(self.PATH_CORRELATION)


        ignore_cols_corr = ['date']
        
        methods = ["pearson", "spearman"]
        cols = [i for i in df.columns if (i not in targets) and (i not in ignore_cols_corr) ]
        n = len(cols)
        n_feats = 15

        print(f"Plotting Correlations for {name}")        
        
        
        plt.figure(figsize=(20,18))
        for method in methods:
            print('Method:', method)
            for i in tqdm(range(n // n_feats)):
                if pval:
                    p_values = self.calculate_pvalues(df = df.loc[: , targets + cols[n_feats * i : n_feats * (i+1)] ], method = method)
                    correlation = df.loc[: , targets + cols[n_feats * i : n_feats * (i+1)] ].corr(method = method, numeric_only=True)
                    mask = np.invert(p_values < 0.05)
                    mask = mask[correlation >= 0.1]
                    sns.heatmap(correlation, mask = mask, annot=True, annot_kws={"size": 14})
                else:
                    correlation = df.loc[: , targets + cols[n_feats * i : n_feats * (i+1)] ].corr(method = method, numeric_only=True)
                    mask = correlation[correlation >= 0.1]
                    sns.heatmap(mask, annot=True, annot_kws={"size": 14})

                plt.title(f"{method.capitalize()} correlation between features and offset, {len(df)} scans")
                plt.tight_layout()
                plt.savefig(self.PATH_CORRELATION + f"Correlation_{method}_{name}_{i}.png")
                plt.clf()

        # Remaining features
            if n % n_feats != 0:
                last_i = n // n_feats
                if pval:
                    p_values = self.calculate_pvalues(df = df.loc[: , targets + cols[n_feats * last_i:]], method = method)
                    mask = np.invert(p_values < 0.05)
                    correlation = df.loc[: , targets + cols[n_feats * last_i:] ].corr(method = method, numeric_only=True)
                    mask = mask[correlation >= 0.1]
                    sns.heatmap(correlation, mask = mask, annot=True, annot_kws={"size": 14})
                else:
                    correlation = df.loc[: , targets + cols[n_feats * last_i:] ].corr(method = method, numeric_only=True)
                    mask = correlation[correlation >= 0.1]
                    sns.heatmap(mask, annot=True, annot_kws={"size": 14})

                plt.title(f"{method.capitalize()} correlation between features and offset, {len(df)} scans")
                plt.tight_layout()
                plt.savefig(self.PATH_CORRELATION + f"Correlation_{method}_{name}_{last_i}.png")
                plt.clf()
        
        plt.close('all')
        print(self.PATH_CORRELATION + f"Correlation_{method}_{name}_{last_i}.png")
        print('Done plotting correlation')

        return

    def plot_pairs(self, name=None, df = None, targets = ["Offset", 'Off_Az_new', 'Off_El_new']):
        if name is None:
            name = self.name
        
        if df is None:
            df = self.df_merged

        if not os.path.exists(self.PATH_PAIRS):
            os.makedirs(self.PATH_PAIRS)

        cols = [i for i in df.columns if i not in targets]
        n = len(cols)
        n_feats = 5
        
        print(f'Plotting pairs for {name}')
        plt.figure(figsize=(20,18))
        for target in targets:
            print('Target:', target)
            for i in tqdm(range(n // n_feats)):
                plt.title( f"Correlation between features and corrections, {len(df)} scans" )
                sns.pairplot(df, hue= "Hour", vars=[target] + cols[n_feats*i:n_feats*(i+1)], palette="BrBG")
                plt.savefig(self.PATH_PAIRS + f"Pairplot_{name}_{target}_{i}.png",dpi = 400)
                plt.clf()

            # Remaining features
            if n % n_feats != 0:
                last_i = n // n_feats
                plt.figure( figsize=(20,18) )
                plt.title( f"Correlation between features and corrections, {len(df)} scans" )
                sns.pairplot(df, hue= "Hour", vars=[target] + cols[n_feats*last_i:], palette="BrBG")
                plt.savefig(self.PATH_PAIRS + f"Pairplot_{name}_{target}_{last_i}.png",dpi = 400)
                plt.clf()


        plt.close('all')

        print('Done plotting pairs')
        
        return
    
    def scan_locations(self):

        if not os.path.exists('./ScanLocation/'):
            os.makedirs('./ScanLocation/')

        cols = ['Offset'] # one or more
        
        fig, ax = plt.subplots(1,1, figsize=(20,10))
        ax.set_xlabel('Az')
        ax.set_ylabel('El')
        ax.set_ylim(0,95)
        ax.axis('equal')
        #plt.colorbar(sc)
        
        if self.patch is not None:
            ax.scatter(self.df_unfiltered['ACTUALAZ'].values, self.df_unfiltered['ACTUALEL'].values, cmap='viridis')#, c=self.df_unfiltered['Offset'].values)
            ax.scatter(self.df_merged['ACTUALAZ'].values, self.df_merged['ACTUALEL'].values)
            n_scans = len(self.df_merged)
            plt.title(f'Scan locations, {n_scans}/{len(self.df_unfiltered)} of all scans')
            plt.savefig(f'./ScanLocation/scan_location_patch.png')
        
        else:
            sc = ax.scatter(self.df_merged['ACTUALAZ'].values, self.df_merged['ACTUALEL'].values, cmap='viridis')
            plt.title(f'Scan locations')
            plt.savefig(f'./ScanLocation/scan_location.png')
            n_scans = len(self.df_merged)

        plt.clf()
        print('Number of scans in patch:', n_scans)




def analysis_pipeline(path, filter_patch):
    """
    Pipeline for analysis of the data
    """
    a = Analysis(path, filter_patch)
    a.plot_correlation()
    return
    a.plot_pairs()

def run_all(path_raw, dt0=None, dt1=None, operation=None, filter_patch = None):

    cols_operation = {
        'median':   ['DEWPOINT', 'HUMIDITY', 'PRESSURE', 'TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'TEMP5', 'TEMP6', 'TEMP26', 'TEMP27', 'TEMP28', 'TEMPERATURE', 'WINDSPEED'],
        'variance': ['ACTUALAZ', 'ACTUALEL', 'POSITIONX', 'POSITIONY', 'POSITIONZ', 'ROTATIONX', 'ROTATIONY', 'ROTATIONZ', 'TILT1X', 'TILT1Y', 'WINDSPEED'],
        'sum_dabs':     ['ACTUALAZ', 'ACTUALEL', 'POSITIONX', 'POSITIONY', 'POSITIONZ', 'ROTATIONX', 'ROTATIONY', 'ROTATIONZ', 'TILT1X', 'TILT1Y', 'WINDSPEED']

    }

    params = {
        1: {'operation': 'median', 'dt0': 2, 'dt1': 2},
        #2: {'operation': 'variance', 'dt0': 2, 'dt1': 2},
        #3: {'operation': 'sum_dabs', 'dt0': 5, 'dt1': 0},
        4: {'operation': 'last', 'dt0': 2, 'dt1':2},
        5: {'operation': 'median', 'dt0': 5, 'dt1': 0},
        #6: {'operation': 'variance', 'dt0': 5, 'dt1': 0}

    }

    for key,obj in params.items():
        operation = obj['operation']
        dt0 = obj['dt0']
        dt1 = obj['dt1']

        path_processed_data = f'./Data/scans_{operation}_dt_{dt0}_{dt1}.csv'
        if not os.path.exists(path_processed_data):
            raw_data_pipeline(path_raw, dt0, dt1, operation)
        analysis_pipeline(path_processed_data, filter_patch)





"""
Offset: [TEMP1-TEMP6, TEMP26-TEMP28, TEMPERATURE, Az_sun, SunAzDiff, TILT1T, POSITIONY]
Off_Az: [SunAngleDiff, SunAngleDiff_15, WINDDIR DIFF, El_sun, SunElDiff, TURBULENCE]
Off_El: [HUMIDITY, Az_sun, El_sun, SunAngleDiff, SunAngleDiff_15, POSITIONX, POSITIONY, PRESSURE]
temps = ['TEMP1','TEMP2','TEMP3','TEMP4','TEMP5','TEMP6','TEMP26','TEMP27','TEMP28','TEMPERATURE','TILT1T']
df = pd.read_csv(path_merged)
df = df.loc[:, df.columns.isin(temps)]

plt.figure(figsize=(20,18))
plt.title(f"Correlation of different temperature measurements")
sns.heatmap(df.corr(method = 'pearson'), annot = True, annot_kws={'size': 14})

plt.tight_layout()
plt.savefig('./Correlation/Temperature.png')
plt.clf()
"""
selected_features = ['ACTUALAZ', 'ACTUALEL', 'TEMP1', 'TEMP26', 'TEMP28', 'TILT1T', 'Az_sun', 'El_sun', 'SunElDiff', 'SunAzDiff',
                    'SunAngleDiff', 'SunAngleDiff_15', 'POSITIONX', 'POSITIONY', 'PRESSURE', 'HUMIDITY', 'WINDDIR DIFF', 'TURBULENCE', 'Hour']



def plot_sun():

    if not os.path.exists('./SunPlots'):
        os.mkdir('./SunPlots')
    df	     = pd.read_csv('./Data/sun_features_15_3.csv')
    df_scans = pd.read_csv('./Data/PointingTable.csv')

    df      ['date']     = pd.to_datetime(      df['date']    )
    df_scans['obs_date'] = pd.to_datetime(df_scans['obs_date'])

    dt0 = pd.Timedelta(minutes = 15)
    dt1 = pd.Timedelta(minutes = 3)

    n_sampels = 10
    n_xticks = 10
    idx_samples = [np.random.randint(0, len(df_scans)) for i in range(n_sampels)]
    plt.figure( figsize=(20,18) )
    for idx in idx_samples:
        df_sample = df.loc[ (df['date'] > df_scans.loc[idx, 'obs_date'] - dt0) & (df['date'] < df_scans.loc[idx, 'obs_date'] + dt1) ]
        plt.plot(df_sample['date'].values, df_sample['Az_sun'].values, 'o', label = 'Sun Az')
        plt.plot(df_sample['date'].values, df_sample['ACTUALAZ'].values, 'o', label = 'APEX Az')
        plt.plot(df_sample['date'].values, df_sample['SunAngleDiff'].values, 'o', label = 'Smallest Angle')
        plt.xticks(df_sample['date'].values[0::int(len(df_sample)/n_xticks)], df_sample['date'].dt.strftime('%H:%M:%S').values[0::int(len(df_sample)/n_xticks)])
        plt.xlabel('Time [HH:MM:SS]')
        plt.ylabel('Azimuth [deg]')
        plt.title(f'Sun Azimuth and APEX Azimuth for scan at {df_scans.loc[idx, "obs_date"]}')
        plt.legend()
        plt.savefig(f'./SunPlots/sunAz_{idx}.png')
        plt.clf()




def get_all_scan_vals():

    df_pointing = pd.read_csv(path_pointing)
    df_pointing['obs_date'] = pd.to_datetime(df_pointing['obs_date'])

    time_offset   = pd.Timedelta(seconds = 53)
    scan_duration = pd.Timedelta(seconds = 90)

    for i,t in tqdm(enumerate(df_pointing['obs_date'])):

        start_observing, end_observing = get_scan_flags(t)

        if start_observing is None:
            start_observing = t + time_offset
            end_observing   = start_observing + scan_duration


def check_merged():
    df = pd.read_csv('./Data/merged_features_all.csv')
    df_scans = pd.read_csv(path_pointing)
    print(df.describe())

    

    df_wind = pd.read_csv(path_median_wind)
    df_sun  = pd.read_csv(path_median_sun)
    df_feats = pd.read_csv(path_median)
    #Drop nan in both dfs
    df_wind  = df_wind.dropna()
    df_sun   = df_sun.dropna()
    df_feats = df_feats.dropna()

    #Drop duplicates in all dfs 
    df_wind  = df_wind.drop_duplicates()
    df_sun   = df_sun.drop_duplicates()
    df_feats = df_feats.drop_duplicates()


    #Convert to pd datetime
    df_wind['date']      = pd.to_datetime(df_wind['date'])
    df_sun['date']       = pd.to_datetime(df_sun['date'])
    df_feats['date']     = pd.to_datetime(df_feats['date'])
    df_scans['obs_date'] = pd.to_datetime(df_scans['obs_date'])


    #Length of both dfs
    print('Length of wind', len(df_wind))
    print('Length of sun',  len(df_sun))
    print('Length of feats',len(df_feats))

    #Print the length of unique dates in all dfs
    print('Length of wind', len(df_wind['date'].unique()))
    print('Length of sun',  len(df_sun['date'].unique()))
    print('Length of feats',len(df_feats['date'].unique()))

    #Check how many dates in df_scans are not in the three dfs
    print('Dates in scans not in wind', len(df_scans[~df_scans['obs_date'].isin(df_wind['date'])]))
    print('Dates in scans not in sun',  len(df_scans[~df_scans['obs_date'].isin(df_sun['date'])]))
    print('Dates in scans not in feats',len(df_scans[~df_scans['obs_date'].isin(df_feats['date'])]))

    #Check how many dates in the three dfs are not in the same three
    print('Dates in wind not in sun',   len(df_wind[~df_wind['date'].isin(df_sun['date'])]))
    print('Dates in wind not in feats', len(df_wind[~df_wind['date'].isin(df_feats['date'])]))
    print('Dates in sun not in feats',  len(df_sun[~df_sun['date'].isin(df_feats['date'])]))
    print('Dates in sun not in wind',   len(df_sun[~df_sun['date'].isin(df_wind['date'])]))
    print('Dates in feats not in wind', len(df_feats[~df_feats['date'].isin(df_wind['date'])]))
    print('Dates in feats not in sun',  len(df_feats[~df_feats['date'].isin(df_sun['date'])]))

    #Join the three dfs on date
    df_merged = df_wind.merge(df_sun, on='date').merge(df_feats, on='date')
    print(df_merged.describe())
    


def find_patches():
    pass


def correction_check():
    df = pd.read_csv('./Data/PointingTable.csv')
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    df = df.loc[ : , df.columns.isin(['Az', 'El', 'ca', 'ie', 'Off_Az', 'Off_El', 'rx'])]
    plt.figure(figsize=(20,18))

    for method in ['spearman', 'pearson']:
        sns.heatmap(df.corr(method = method, numeric_only=True), annot=True, annot_kws={"size": 14})
        plt.title(f"Correlation {method}")
        plt.tight_layout()
        #plt.savefig(f"Correlation_correction_check_{method}.png")
        plt.clf()

    print(df['rx'].unique())
    print(df['rx'].value_counts())


def corr_temp():
    """
    Plots correlation for all the temperature instruments
    If it needs to be run again, remember to change the title of the figure
    """
    a = Analysis()
    df = pd.read_csv('./Data/processed_v4/all_features_safe.csv')
    print(df.columns)
    df = df.loc[: , ['TEMP' + str(i) + '_MEDIAN' for i in [1,2,3,4,5,6,26,27,28]] + ['TEMPERATURE_MEDIAN', 'TILT1T_MEDIAN']]

    #Rename all the columns to not have the _MEDIAN suffix
    df.columns = ['TEMP' + str(i) for i in [1,2,3,4,5,6,26,27,28]] + ['TEMPERATURE', 'TILT1T']

    #Plot correlation for all the temperature instrument
    # Compute the correlation matrix
    corr = df.corr()
    plt.figure(figsize=(16,16))
    plt.rcParams['axes.titlesize'] = 25
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Correlation of different temperature sensors')
    plt.tight_layout()
    plt.savefig('Correlation_temp.png')
    
    embed()

def corr_xyz():
    """
    Plots correlation for rotation and position xyz etc.
    If it needs to be run again, remember to change the title of the figure
    """
    a = Analysis()
    df = pd.read_csv('./Data/scans_median_0_0_all.csv')
    targets1 = [i + j for i in ['TILT1', 'ROTATION'] for j in ['X','Y']]
    targets2 = ['POSITION' + i for i in ['X','Y','Z']]
    df = df.loc[: , targets1 + targets2]

    a.plot_correlation('xyz', df, [])


def autocorr(col):
    
    df = pd.read_csv('./Data/PointingTable.csv')
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    df.set_index('obs_date', inplace=True)
    
    #Remove rows where col < -25 and > 25
    df = df.loc[df[col] > -25]
    df = df.loc[df[col] < 25]

    fig, ax = plt.subplots(2)

    ax[0].plot(df[col])

    plot_acf(df[col],ax=ax[1])
    plt.tight_layout()
    plt.savefig(f'{col}.png', dpi = 400)

def boxplots_instruments():
    df = pd.read_csv('./Data/PointingTable.csv')
    df.obs_date = pd.to_datetime(df.obs_date)
    #insert 'Off' as magnitude of off az and off el
    df['Off'] = np.sqrt(df['Off_Az']**2 + df['Off_El']**2)
    #df.loc[: , ['obs_date', 'rx', 'Az', 'El', 'Off', 'Off_Az', 'Off_El', 'ca', 'ie']].to_csv('./Data/PointingTable2.csv', index=False)
    #corr_xyz()
    #remove offsets with abs value > 25
    df = df.loc[df['Off'] < 25]
    df = df.loc[df['Off'] > -25]

    #make the same box plots for az, el and off_az, off_el
    for col in ['Az', 'El', 'Off_Az', 'Off_El', 'Off']:
        fig, ax = plt.subplots(figsize=(20,18))
        sns.set_style('whitegrid')
        ax = sns.boxplot(x='rx', y=col, data=df)
        #add legend with number of observations in each group, and make xlabels smaller
        ax.legend(labels= [f'{i:10s} {count:>5d}' for i,count in zip(df['rx'].value_counts().index, df['rx'].value_counts().values)], title='Number of observations')
        plt.savefig(f'{col}_boxplot.png')
        plt.clf()

def get_scan_flags_modified(datetime, index, df_pointing, duplicates, dir_list):

    date = str(datetime.date())

    if date in dir_list:
        df_dump = pd.read_csv(f'{path_tiltmeter_dumps}/Tiltmeter_{date}.dump', names = ['date','A', 'B', 'C', 'D', 'E', 'flag'])
        df_dump['date'] = pd.to_datetime(df_dump['date'])
        df_dump_interval = df_dump.loc[(df_dump['date'] > datetime - pd.Timedelta(minutes = 5)) & (df_dump['date'] < datetime + pd.Timedelta(minutes = 15))]



        if len(df_dump_interval) > 0:

            # if the last date in df_dump_interval is witin 5 min of midnight, read the next day's dump file and concat the dataframe
            if df_dump_interval['date'].iloc[-1].hour == 23 and df_dump_interval['date'].iloc[-1].minute >= 55:
                df_dump2 = pd.read_csv(f'{path_tiltmeter_dumps}/Tiltmeter_{str(datetime.date() + pd.Timedelta(days=1))}.dump', names = ['date','A', 'B', 'C', 'D', 'E', 'flag'])
                df_dump2['date'] = pd.to_datetime(df_dump2['date'])
                df_dump_interval = pd.concat([df_dump_interval, df_dump2.loc[df_dump2['date'] < datetime + pd.Timedelta(minutes = 15)]])

            #For duplicate obs_date, have to check if it is the first or the last one in PointingTable
            if datetime in duplicates and index == df_pointing[df_pointing['obs_date'] == datetime].index.max():
                first = False
            else:
                first = True


            df_dump_interval.loc[ df_dump_interval['flag'] == 'IDLE'     , 'flag'] = 0
            df_dump_interval.loc[ df_dump_interval['flag'] == 'PREPARING', 'flag'] = 0
            df_dump_interval.loc[ df_dump_interval['flag'] == 'OBSERVING', 'flag'] = 4
            df_dump_interval.insert(0, 'dt', (df_dump_interval['date'] - datetime) )
            df_dump_interval.loc[:, 'dt'] = df_dump_interval.loc[ df_dump_interval['dt'] > pd.Timedelta(0), 'dt' ]
            diff = df_dump_interval['flag'].diff()
            #print(f'Scan time: {datetime}')
            #print('Start preparing')
            #print(df_dump_interval.loc[diff == 1, ['date', 'flag'] ])
            #start_preparing = df_dump_interval.loc[ (df_dump_interval.loc[:, 'dt' ] == df_dump_interval.loc[diff == 1, 'dt' ].min()) & (diff == 1) ] 
            #print('Start observing')
            #print(df_dump_interval.loc[diff == 4, ['date', 'flag'] ])
            #start_observing = df_dump_interval.loc[ (df_dump_interval.loc[:, 'dt' ] == df_dump_interval.loc[diff == 4, 'dt' ].min()) & (diff == 4) ]
            start_observing = df_dump_interval.loc[ diff == 4 ].dropna()
            
            # print('----------', datetime, '-----------')
            # print('---------- Start Observing ----------')
            # print(start_observing)
            #print('End observing')
            #print(df_dump_interval.loc[diff == -4, ['date', 'flag'] ])
            if index in [2280, 2281, 2282, 2283, 2284,2285,2286]:
                print(start_observing)
                embed()
            if start_observing.empty:
                return None, None
            
            elif len(start_observing) == 1 and first is False:
                return None, None
            
            if first:
                start_observing = start_observing['date'].iloc[0]
            else:
                try:
                    start_observing = start_observing['date'].iloc[1]
                except:
                    print(index, datetime)
                    embed()
            end_observing = df_dump_interval.loc[ (df_dump_interval['date'] > start_observing) & (diff == -4) ]
            end_observing = end_observing.loc[ end_observing['date'] > start_observing]
            # print('---------- End Observing ----------')
            # print(end_observing)

            if end_observing.empty:
                return None, None
            
            end_observing   = end_observing['date'].iloc[0]

            #print(start_observing, end_observing)
            return start_observing, end_observing

        else:
        #print(f'No data for date {date}')
           return None, None
    
    else:
        #print(f'No data for date {date}')
        return  None, None

def check_duplicate_datetime():

    df = pd.read_csv('./Data/PointingTable2.csv')
    df.obs_date = pd.to_datetime(df.obs_date)

    t = df.obs_date
    # find duplicated times
    print(t.describe())
    t = t[t.duplicated()]
    print(t.describe())
    duplicates = t.to_list()
    dir_list = os.listdir(path_tiltmeter_dumps)
    dir_list = [i.split('.')[0].split('_')[1] for i in dir_list]
    df_times = pd.DataFrame(columns = ['obs_date', 'index', 'start', 'end'])

    t = df.obs_date
    for index, datetime in t.iloc[2000:].items():
        start, end = get_scan_flags_modified(datetime, index, df, duplicates, dir_list)
        _row = {'obs_date': datetime, 'index': index, 'start': start, 'end': end}
        df_times = pd.concat([df_times, pd.DataFrame(_row, columns = df_times.columns, index = [index])], ignore_index = True)
    
    print(df_times)
    embed()
    df_times = df_times.dropna()
    #check that start is after obs_date and end is after start for all scans
    df_times['start_after_obs'] = df_times['start'] > df_times['obs_date']
    df_times['end_after_start'] = df_times['end'] > df_times['start']
    embed()
    #check for scans with duplicate obs_date, that the scan with the highest index has start after the end of the scan with the lowest index
    #Keep only duplicates in obs_date
    df_times = df_times[df_times['obs_date'].duplicated()]
    df_times['start_after_end'] = df_times['start'] > df_times['end'].shift(-1)
    embed()

def analyze_tiltmeter():
    date = '2022-05-21'
    df_tilt = pd.read_csv(f'{path_tiltmeter_dumps}/Tiltmeter_{date}.dump', names = ['date','A', 'B', 'C', 'D', 'E', 'flag'])
    df_tilt['date'] = pd.to_datetime(df_tilt['date'])
    t1 = pd.Timestamp(2022, 5, 21, 23)
    df_tilt = df_tilt[df_tilt['date'] > t1]

    df_tilt.loc[ df_tilt['flag'] == 'IDLE'     , 'flag'] = 0
    df_tilt.loc[ df_tilt['flag'] == 'PREPARING', 'flag'] = 0
    df_tilt.loc[ df_tilt['flag'] == 'OBSERVING', 'flag'] = 4
    diff = df_tilt['flag'].diff()
    start_observing = df_tilt.loc[ diff == 4 ].dropna()
    

    df_pointing = pd.read_csv('./Data/PointingTable2.csv')

    embed()

def scan_duration_distribution():   
    """
    Finds values for all scan labels, regardsless of observing flag or not.

    Mean duration: 121.5 seconds
    Mean difference between timestamp and actual scan: 53 seconds
    """

    df_pointing = pd.read_csv('./Data/tmp2022_cleanedRules.csv')
    # df_pointing['obs_date'] = pd.to_datetime(df_pointing['obs_date'])
    """
    date = []
    start = []
    end   = []
    diff     = []
    duration = []
    rx = []
    for i,t in tqdm(enumerate(df_pointing['obs_date'])):
        start_observing, end_observing = get_scan_flags(t)

        if start_observing is None:
            continue
        diff.append(t-start_observing)
        duration.append(end_observing - start_observing)
        date.append(t)
        rx.append(df_pointing['rx'].iloc[i])

        start.append(start_observing)
        end.append(end_observing)
    

    duration = duration / np.timedelta64(1, 's')
    duration = np.array(duration)
    
    diff     = np.array(diff)
    diff     = - diff / pd.Timedelta('1s')

    #save the lists to file as pandas df
    df = pd.DataFrame({'date': date, 'start': start, 'end': end, 'diff': diff, 'duration': duration, 'rx': rx})
    df.to_csv('./Data/scan_durations2022.csv', index = False)

    print(f"Mean diff: {np.mean(diff)}")
    print(f"Mean duration: {np.mean(duration)}")
    """
    #print number of values of 300 seconds in diff
    df = pd.read_csv('./Data/scan_durations2022.csv')
    df_pointing['filter'] = df_pointing.rx + '#' + df_pointing.obs_date
    df['filter'] = df.rx + '#' + df.date

    df = df[df['filter'].isin(df_pointing['filter'])]
    df['date'] = pd.to_datetime(df['date'])
    """embed()
    sns.set(rc = {'figure.figsize':(20,10)})
    fig, ax = plt.subplots(1,2)
    sns.histplot(data = df, x = 'duration', hue = 'rx', ax = ax[0])
    sns.histplot(data = df.loc[ df['diff']], x = 'diff', hue = 'rx', ax = ax[1], binrange = (50,110))
    ax[0].set_title('Duration of scan')
    ax[1].set_title('Difference between timestamp and actual scan')
    fig.savefig(f'./Figures/scan_duration_distribution.png')
    """
    fontsize = 22
    plt.rcParams['axes.titlesize'] = 24; plt.rcParams['axes.labelsize'] = fontsize;
    plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = fontsize; plt.rcParams["legend.fontsize"] = fontsize
    #make a boxplot of the duration of scans with different rx
    fig, ax = plt.subplots(1,2,figsize=(25, 10))
    sns.boxplot(data = df, x = 'rx', y = 'duration', ax = ax[0])
    sns.boxplot(data = df, x = 'rx', y = 'diff', ax = ax[1])
    ax[0].set_title('Duration of pointing scans')
    ax[1].set_title('Difference between timestamp and start of pointing scans')
    ax[0].set_title('Duration of pointing scans')
    ax[0].set_xlabel('Instrument')
    ax[0].set_ylabel('Duration [seconds]')
    ax[0].set_ylim([0,350])
    
    ax[1].set_title('Difference between timestamp and start of pointing scans')
    ax[1].set_xlabel('Instrument')
    ax[1].set_ylabel('Time difference [seconds]')
    ax[1].set_ylim([0,100])
    
    plt.tight_layout()
    fig.savefig(f'./Figures/scan_duration_distribution_rx.pdf')
    

    #print the mean duration for each of the rx using groupby in the df
    print(df.groupby('rx').mean())

    #plot scan duration and diff as a function of date, with different rx as different colors, also add linear regression line
    fig, ax = plt.subplots(1,2,figsize=(18, 10))

    #Change the size of font to be smaller
    plt.rcParams['axes.titlesize'] = 18
    sns.scatterplot(data = df, x = 'date', y = 'duration', hue = 'rx', ax = ax[0])
    sns.scatterplot(data = df, x = 'date', y = 'diff', hue = 'rx', ax = ax[1])


    fig.savefig(f'./Figures/scan_duration_distribution_date.pdf')



    embed()



    plt.clf()

    # Create subplots
    fig, ax = plt.subplots(1, 2,figsize=(18, 10))

    # Plot the first subplot
    sns.scatterplot(data=df, x='date', y='duration', hue='rx', ax=ax[0])
    ax[0].set_xlabel('Month in 2022')
    ax[0].set_ylabel('Duration [seconds]')
    ax[0].set_title('Duration of pointing scans')
    ax[0].xaxis.set_major_locator(mdates.MonthLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax[0].legend_.remove()
    ax[0].set_ylim([0,350])


    # Plot the second subplot
    sns.scatterplot(data=df, x='date', y='diff', hue='rx', ax=ax[1])
    ax[1].set_ylim([0,350])
    ax[1].set_xlabel('Month in 2022')
    ax[1].set_ylabel('Time difference [seconds]')
    ax[1].set_title('Difference between timestamp and start of pointing scans')
    ax[1].xaxis.set_major_locator(mdates.MonthLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    fig.savefig('./Figures/scan_duration_distribution_date.pdf', dpi = 300)


def data_frequency_during_scans():
    
    df_pointing = pd.read_csv('./Data/PointingTable2.csv')
    df_pointing['obs_date'] = pd.to_datetime(df_pointing['obs_date'])
    df_pointing = df_pointing.rename(columns = {'obs_date': 'date'})
    df_raw = pd.read_csv('./Data/raw/raw_data_15_10.csv')
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    df_scanDurations = pd.read_csv('./scan_durations.csv')
    df_scanDurations['date'] = pd.to_datetime(df_scanDurations['date'])

    timeBefore = pd.Timedelta('1m')
    timeAfter = pd.Timedelta('0m')

    for i, timestamp in df_pointing['date'].items():

        if i == 0:
            count = df_raw[(df_raw['date'] > timestamp - timeBefore) & (df_raw['date'] < timestamp + timeAfter)].count()
        else:
            count += df_raw[(df_raw['date'] > timestamp - timeBefore) & (df_raw['date'] < timestamp + timeAfter)].count()
        

    count /= len(df_pointing['date'])
    count /= (timeBefore + timeAfter) / pd.Timedelta('1m') #minutes
    print(count)

    embed()

def data_frequency_processed_data(path = None):
    #Reads the proccessed data in './Data/processed_v2/' and plots the number of non nan observations in each column

    path = './Data/db_exports/'
    filenames = os.listdir(path)

    freqs = {}
    for fn in tqdm(filenames):
        col = fn.split('.')[0]

        df = pd.read_csv(path + fn)
        df['date'] = pd.to_datetime(df['date'])

        df = df[df.date >= pd.Timestamp('2022-05-18 17:00:00')]
        df = df[df.date < pd.Timestamp('2022-05-18 17:30:00')]
        frequency = 60 / (df.count()/((df.date.iloc[-1]-df.date.iloc[0])/pd.Timedelta('1m')))
        freqs[col] = frequency[col]
        print(col, frequency[col])
    embed()
    # pd.DataFrame(data = freqs).to_csv('./Data/data_frequency.csv', index = False)
    return 

"""
rx         duration       diff
HOLO       41.952381   26.809524
LASMA345   56.034335  144.781116
NFLASH230  50.996109  101.913100
NFLASH460  53.078498  103.436860
SEPIA180   52.660131  191.810458
SEPIA345   54.386555  145.500000
SEPIA660   55.280193  123.623188
"""



def cluster_bad_scans():

    df = pd.read_csv('./Data/PointingTable.csv')
    df['beamsize'] = 7.8 * 800 / df.freq

    df = df[(df.Off_El < 70) & (df.Off_Az < 70)]
    df = df[(df.Off_eEl < 70) & (df.Off_eAz < 70)]
    signal_cols = ['Amp_Az', 'Amp_El', 'FWHM_Az', 'FWHM_El', 'Off_Az', 'Off_El']
    noise_cols =  ['Amp_eAz', 'Amp_eEl', 'FWHM_eAz', 'FWHM_eEl', 'Off_eAz', 'Off_eEl']

    for signal,noise in zip(signal_cols, noise_cols):
        df[signal] = df[signal] / df['beamsize']
        df[noise] = df[noise] / df['beamsize']

    #Signal to noise ratio
    df['Amp_rAz'] = df.Amp_Az / df.Amp_eAz
    df['Amp_rEl'] = df.Amp_El / df.Amp_eEl
    df['FWHM_rAz'] = df.FWHM_Az / df.FWHM_eAz
    df['FWHM_rEl'] = df.FWHM_El / df.FWHM_eEl
    df['Off_rAz'] = df.Off_Az / df.Off_eAz
    df['Off_rEl'] = df.Off_El / df.Off_eEl



    #Quality merit
    df['QM_Az'] = np.sqrt(df.Amp_rAz**2 + df.FWHM_rAz**2 + df.Off_rAz**2)
    df['QM_El'] = np.sqrt(df.Amp_rEl**2 + df.FWHM_rEl**2 + df.Off_rEl**2)

    df['QM_Az2'] = df.Amp_rAz * df.FWHM_rAz * df.Off_rAz
    df['QM_El2'] = df.Amp_rEl * df.FWHM_rEl * df.Off_rEl


    #Replacing -inf and inf with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    embed()
    #plot histograms of the different quality metrics
    fig, ax = plt.subplots(2,2)
    
    ax[0,0].hist(df.QM_Az,  bins = 30, label='QM_Az')
    ax[0,1].hist(df.QM_El,  bins = 30, label='QM_El')
    ax[1,0].hist(df.QM_Az2, bins = 30, label='QM_Az2')
    ax[1,1].hist(df.QM_El2, bins = 30, label='QM_El2')
    plt.savefig('./Figures/QualityMetrics.png')


    """
    fig, ax = plt.subplots(2,2)
    sns.histplot(data = df, x = 'QM_Az', ax = ax[0,0])
    sns.histplot(data = df, x = 'QM_El', ax = ax[0,1])
    sns.histplot(data = df, x = 'QM_Az2', ax = ax[1,0])
    sns.histplot(data = df, x = 'QM_El2', ax = ax[1,1])
    plt.savefig('./Figures/QualityMetrics.png')
    plt.clf()
    return
    embed()
    """

    pp_cols = ['Off_Az', 'Off_El', 'QM_Az', 'QM_El', 'QM_Az2', 'QM_El2']
    #Make 4 different pairplots of the data in signal_cols and the different quality metrics as hue
    sns.pairplot(df, vars = pp_cols, hue = 'QM_Az', palette = 'viridis')
    plt.savefig('./Figures/Pairplot_QM_Az.png')
    plt.clf()
    sns.pairplot(df, vars = pp_cols, hue = 'QM_El', palette = 'viridis')
    plt.savefig('./Figures/Pairplot_QM_El.png')
    plt.clf()
    sns.pairplot(df, vars = pp_cols, hue = 'QM_Az2', palette = 'viridis')
    plt.savefig('./Figures/Pairplot_QM_Az2.png')
    plt.clf()
    sns.pairplot(df, vars = pp_cols, hue = 'QM_El2', palette = 'viridis')
    plt.savefig('./Figures/Pairplot_QM_El2.png')
    plt.clf()

y_labels = {
    'ACTUALAZ': 'Azimuth [radians]',
    'ACTUALEL': 'Elevation [radians]',
    'COMMANDAZ': 'Azimuth [radians]',
    'COMMANDEL': 'Elevation [radians]',
    'TEMP1': 'Temperature [$^\circ C$]',
    'TILT1T': 'Temperature [$^\circ C$]',
    'TEMP26': 'Temperature [$^\circ C$]',
    'WINDSPEED': 'Wind speed [m/s]',
    'WINDDIRECTION': 'Wind direction [$^\circ$]',
}

def plotScanFeature_v2(features):

    if not os.path.exists('./ScanFeatures'):
        os.makedirs('./ScanFeatures')

    dfs = [pd.read_csv(f'./Data/db_exports/{f}.csv') for f in features]

    for df in dfs:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
    
    df_scans = pd.read_csv('./Data/tmp2022_cleanedRules.csv')
    df_scans.rename(columns={'obs_date': 'date'}, inplace=True)
    df_scans['date'] = pd.to_datetime(df_scans['date'])
    df_scans.sort_values(by='date', inplace=True)

    n_sampels = 150
    idx_samples = [np.random.randint(0, len(df_scans)) for i in range(n_sampels)]
    #idx_samples = [i for i in range(len(df_scans))]

    count = 0
    max_count = 5
    for i in (idx_samples):

        start_observing, end_observing = get_scan_flags(df_scans['date'].iloc[i])
        #print(i, df_scans['date'].iloc[i], a, b, c)
        #print(start_observing, end_observing)
        if start_observing is None:
            continue
        else:
            pass
        
        #print(f'--- Sample index {i}, date  ---')

        sample = df_scans.iloc[i]

        df_samples = [df.loc[(df['date'] >= sample['date'] - pd.Timedelta(minutes = 5)) & (df['date'] <= sample['date'] + pd.Timedelta(minutes = 5))] for df in dfs]
        #print(df_sample.describe())
        
        
        for df_sample,feature in zip(df_samples,features):
            fig, ax = plt.subplots(figsize=(10,6))
            if not os.path.exists(f'./ScanFeatures_v2/{feature}/'):
                os.makedirs(f'./ScanFeatures_v2/{feature}/')

            # fig.set_size_inches(18, 16)
            # fig.suptitle(f'{feature} during scan at ' + str(sample['date']))

            ax.ticklabel_format(style='plain', axis='y')
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            
            
            # ax.set_title(f'Offsets from scan | Total offset: {np.sqrt(df_scans.Off_El.iloc[i]**2 + df_scans.Off_Az.iloc[i]**2):.2f} | Off_El: {df_scans.Off_El.iloc[i]:.2f} | Off_Az: {df_scans.Off_Az.iloc[i]:.2f}')
            ax.set_title(f'{feature} during pointing scan at {str(sample["date"])}')
            ax.set_ylabel(y_labels[feature])
            ax.set_xlabel('Time [HH:MM]')
            ax.plot(df_sample.loc[ (df_sample['date'] < start_observing) | (df_sample['date'] >= end_observing), 'date'].values, df_sample.loc[ (df_sample['date'] < start_observing) | (df_sample['date'] >= end_observing), feature].values, 'bo', label = 'Not observing')
            ax.plot(df_sample.loc[ (df_sample['date'] >= start_observing) & (df_sample['date'] < end_observing), 'date'].values, df_sample.loc[ (df_sample['date'] >= start_observing) & (df_sample['date'] < end_observing), feature].values, 'ro', label = 'Observing')
            ax.axvline(sample['date'], color = 'r', label = 'Scan timestamp')
            #Fix the xlabel to only show time as hh:mm:ss format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.tight_layout()
            ax.legend()

            #plt.tight_layout(pad=1, w_pad=1, h_pad=2.0, )
            plt.savefig(f'./ScanFeatures_v2/{feature}/scan_{feature}_{i}.pdf', dpi = 300)
            plt.close(fig)

        count += 1

        plt.close(fig)
        if count == max_count:
            break

    return

def plotScanFeature_optical(features):

    path_save = './ScanFeatures_optical/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    dfs = [pd.read_csv(f'./Data/db_exports/{f}.csv') for f in features]

    for df in dfs:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
    
    df_scans = pd.read_csv('./Data/raw_nflash230.csv')
    df_scans['date'] = pd.to_datetime(df_scans['date'])
    df_scans.sort_values(by='date', inplace=True)
    df_scans = df_scans[df_scans.date < pd.Timestamp('2022-09-17')]
    n_sampels = 150
    idx_samples = [np.random.randint(0, len(df_scans)) for i in range(n_sampels)]
    #idx_samples = [i for i in range(len(df_scans))]

    count = 0
    max_count = 10
    for i in (idx_samples):

        sample = df_scans.iloc[i]

        df_samples = [df.loc[(df['date'] >= sample['date'] - pd.Timedelta(minutes = 5)) & (df['date'] <= sample['date'] + pd.Timedelta(minutes = 5))] for df in dfs]
        #print(df_sample.describe())
        
        
        for df_sample,feature in zip(df_samples,features):
            fig, ax = plt.subplots(figsize=(10,6))
            path_feature = os.path.join(path_save, feature)
            if not os.path.exists(path_feature):
                os.makedirs(path_feature)

            # fig.set_size_inches(18, 16)
            # fig.suptitle(f'{feature} during scan at ' + str(sample['date']))

            ax.ticklabel_format(style='plain', axis='y')
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            
            
            # ax.set_title(f'Offsets from scan | Total offset: {np.sqrt(df_scans.Off_El.iloc[i]**2 + df_scans.Off_Az.iloc[i]**2):.2f} | Off_El: {df_scans.Off_El.iloc[i]:.2f} | Off_Az: {df_scans.Off_Az.iloc[i]:.2f}')
            ax.set_title(f'{feature} during optical scan at {str(sample["date"])}')
            ax.set_ylabel(feature)
            ax.set_xlabel('Time [HH:MM]')
            ax.plot(df_sample['date'].values, df_sample[feature].values, 'bo')
            ax.axvline(sample['date'], color = 'r', label = 'Scan timestamp')
            #Fix the xlabel to only show time as hh:mm:ss format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.tight_layout()
            ax.legend()

            #plt.tight_layout(pad=1, w_pad=1, h_pad=2.0, )
            plt.savefig(os.path.join(path_feature, f'scan_{feature}_{i}.pdf'), dpi = 300)
            plt.close(fig)

        count += 1

        plt.close(fig)
        if count == max_count:
            break

    return


from move_img import move_img
def move_random_imgs():

    df = pd.read_csv('./Data/tmp2022.csv')

    #sample 50 random scans from df.scan and add them to list
    
    #idx_samples = random.sample(sorted(df.scan.unique()), 50)
    df['bs'] = 7.8*800/df.freq
    df = df[(df.Off_eEl > df.bs*0.8) | (df.Off_eAz > df.bs*0.8) | (df.Off_Az > df.bs*0.8) | (df.Off_El > df.bs*0.8)]

    idx_samples = df.scan.unique()

    for i in idx_samples:
        move_img(str(i))


def correlation_with_csv():
    df_scans = pd.read_csv('./Data/tmp2022_clean.csv')
    df_scans.rename(columns={'obs_date': 'date'}, inplace=True)
    df_scans['date'] = pd.to_datetime(df_scans['date'])
    df_scans.sort_values(by='date', inplace=True)

    df_features = pd.read_csv('./Data/processed_v2/integrated_pos.csv')
    df_features['date'] = pd.to_datetime(df_features['date'])

    print(len(df_features), len(df_features.dropna()))
    
    df_merged  = pd.merge(df_scans[['date', 'Off_El', 'Off_Az']], df_features, on='date', how='inner')

    embed()

    #plot correlation of df_merged using seaborn heatmap
    fig, ax = plt.subplots(figsize=(18, 16))
    fig.set_size_inches(18, 16)
    fig.suptitle(f'Correlation of Offsets with features')

    sns.heatmap(df_merged.corr(method = 'spearman'), annot=True, cmap='coolwarm', ax=ax)
    plt.tight_layout(pad=1, w_pad=1, h_pad=2.0, )
    plt.savefig(f'./Correlation_v3/Correlation_temps_pos.png')
    plt.close(fig)

def df_correlation(df, save_path, name):
    correlated_feature = []
    # plot correlation of df_merged using seaborn heatmap
    fig, ax = plt.subplots(figsize=(25, 18))
    fig.set_size_inches(25, 18)
    correlated_features_az_pearson = []
    correlated_features_az_spearman = []
    correlated_features_el_pearson = []
    correlated_features_el_spearman = []
    correlated_features_unscaled_pearson = []
    correlated_features_unscaled_spearman = []
    for method in ['pearson', 'spearman']:
        fig.suptitle(f'Correlation of Offsets with features')
        corr = df.corr(method=method)
        # sort by correlation to column 1
        sorted_corr_1 = corr.iloc[:, 0].sort_values(ascending=False)[1:]
        # print correlations to column 1
        print(f"{method} correlation to azimuth offset (sorted):")
        count = 0
        for col, val in sorted_corr_1.items():
            if np.abs(val) >= 0.1:
                print(f"{col}: {val:.3f}")
                count += 1
                correlated_feature.append(col)
                if method == 'pearson':
                    correlated_features_az_pearson.append(col)

                if method == 'spearman':
                    correlated_features_az_spearman.append(col)
                
        print(f'Number of features with correlation > 0.1: {count}')
        print()
        # sort by correlation to column 2
        sorted_corr_2 = corr.iloc[:, 1].sort_values(ascending=False)[2:]
        # print correlations to column 2
        print(f"{method} correlation to elevation offset (sorted):")
        count = 0
        for col, val in sorted_corr_2.items():
            if np.abs(val) >= 0.1:
                print(f"{col}: {val:.3f}")
                count += 1
                correlated_feature.append(col)
                if method == 'pearson':
                    correlated_features_el_pearson.append(col)

                if method == 'spearman':
                    correlated_features_el_spearman.append(col)

        print(f'Number of features with correlation > 0.1: {count}')
        print()

        sorted_corr_3 = corr.iloc[:, 2].sort_values(ascending=False)[3:]


        print(f"{method} correlation to unscaled azimuth offset (sorted):")
        count = 0
        for col, val in sorted_corr_3.items():
            if np.abs(val) >= 0.1:
                print(f"{col}: {val:.3f}")
                count += 1
                correlated_feature.append(col)
                if method == 'pearson':
                    correlated_features_unscaled_pearson.append(col)

                if method == 'spearman':
                    correlated_features_unscaled_spearman.append(col)


        print(f'Number of features with correlation > 0.1: {count}')
        print()
        # # Generate a mask for the upper triangle
        # mask = np.triu(np.ones_like(corr, dtype=bool))
        # # Set up the matplotlib figure
        # f, ax = plt.subplots(figsize=(25, 18))
        # # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # # Draw the heatmap with the mask and correct aspect ratio
        # annot_corr = np.round(corr, decimals=2)
        # sns.heatmap(annot_corr, mask=mask, cmap=cmap, annot=True, fmt='.2f', annot_kws={"size": 14})
        # plt.xticks(rotation=45, ha='right', fontsize=12)
        # plt.yticks(rotation=0, fontsize=12)
        # plt.savefig(os.path.join(save_path, f'Correlation_{name}_{method}.png'), dpi=300, bbox_inches='tight')
        # plt.clf()

        corr_all = corr[['OFFSETAZ', 'OFFSETEL', 'OFFSETAZ_UNSCALED']]
        corr_all = corr_all[(corr_all.OFFSETAZ >= 0.1) | (corr_all.OFFSETEL >= 0.1) | (corr_all.OFFSETAZ_UNSCALED >= 0.1)]
        
        corr_all.to_latex(f'FinalResultsOptical/correlations_{method}_latex.txt', float_format="%.2f")

    embed()
    plt.close(fig)

    return set(correlated_feature)







def uncorrected_performance():

    """
    Print the actual offsets of the scans.
    Also print the perfomance for no correction at all.
    This is slightly unrealistic as there would be some correction from the 
    analytical model. Therefore, I chose the optimal correction (mean value of offsets)
    and applied that, which results in "meancorr". This would be best benchmark for comparison.
    So a performance of about 6'' would be good, and 3-4 would be even better.
    """
    df = pd.read_csv('./Data/tmp2022_clean_v2.csv')
    
    df = df[['obs_date', 'rx', 'Off_Az', 'Off_El', 'ca', 'ie']]
    df = df[df.rx == 'NFLASH230']
    
    df['az_uncorr'] = df.Off_Az + df.ca
    df['el_uncorr'] = df.Off_El - df.ie
    df['az_meancorr'] = df.az_uncorr - df.ca.mean()
    df['el_meancorr'] = df.el_uncorr + df.ie.mean()
    df['Offset'] = np.sqrt(df.Off_Az**2 + df.Off_El**2)
    df['Offset_uncorr'] = np.sqrt(df.az_uncorr**2 + df.el_uncorr**2)
    df['Offset_meancorr'] = np.sqrt(df.az_meancorr**2 + df.el_meancorr**2)
    print('Performance with and without corrections for clean data.')
    print(df.loc[: , ~df.columns.isin(['obs_date', 'rx'])].mean())

    df = pd.read_csv('./Data/tmp2022.csv')
    
    df = df[['obs_date', 'rx', 'Off_Az', 'Off_El', 'ca', 'ie']]
    df = df[df.rx == 'NFLASH230']
    
    df['az_uncorr'] = df.Off_Az + df.ca
    df['el_uncorr'] = df.Off_El - df.ie
    df['az_meancorr'] = df.az_uncorr - df.ca.mean()
    df['el_meancorr'] = df.el_uncorr + df.ie.mean()
    df['Offset'] = np.sqrt(df.Off_Az**2 + df.Off_El**2)
    df['Offset_uncorr'] = np.sqrt(df.az_uncorr**2 + df.el_uncorr**2)
    df['Offset_meancorr'] = np.sqrt(df.az_meancorr**2 + df.el_meancorr**2)
    print('Performance with and without corrections for all data.')
    print(df.loc[: , ~df.columns.isin(['obs_date', 'rx'])].mean())

def performance_benchmark():
    """
    Prints out the performance of the model with corrections for NFLASH230
    for all(maybe not all) data and clean data. This would be the ideal bencmark
    for a new model.
    """
    df = pd.read_csv('./Data/scans_nflash230_unscaled.csv')
    
    df = df[['Off_Az', 'Off_El']]
    df['Offset'] = np.sqrt(df.Off_Az**2 + df.Off_El**2)
    print('Performance of model with corrections for clean data.')
    print(df.abs().mean())

    df = pd.read_csv('./Data/scans_nflash230_unscaled_all.csv')
    
    df = df[['Off_Az', 'Off_El']]
    df['Offset'] = np.sqrt(df.Off_Az**2 + df.Off_El**2)
    print('Performance of model with corrections for all data.')
    print(df.abs().mean())

def plot_df_corr():
    # performance_benchmark()
    feature_list = []
    n = 1
    save_path = './FinalResultsOptical/'
    dfs = ['dataset_optical_v2']
    ignore_cols = ['DAZ_SPEM_MEDIAN', 'DAZ_TEMP_MEDIAN','DEL_TEMP_MEDIAN','Off_Az','Off_El']
    for fn in dfs:
        df = pd.read_csv(f'./Data/{fn}.csv')
        df = df.loc[: , ~df.columns.isin(ignore_cols)]
        #Make RESIDUALAZ and RESIDUALEL the two first columns
        cols = df.columns.tolist()
        cols1 = cols[-3:]
        cols2 = cols[:-3]
        df = df[cols]
        n_cols = len(cols2)
        chunk_size = n_cols//n + 1
        for i in range(n):
            start = i*chunk_size
            end = (i+1)*chunk_size if i != n-1 else n_cols
            df_subset = df[cols1 + cols2[start:end]]
            correlated_features = df_correlation(df_subset, save_path, f"{fn}_{i+1}")
            feature_list += list(correlated_features)

    print(feature_list)
    embed()

def feature_selection_final(k):
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

    targets = ['Off_Az', 'Off_El']

    datasets = [fn for fn in os.listdir('./Datasets/') if not fn.endswith('.csv')]
    dataset_tests = {i: f'./Datasets/{fn}/features_offsets.csv' for i, fn in enumerate(datasets)}

    for i, fn in dataset_tests.items():
        df_all = pd.read_csv(fn)
        df_all = df_all.dropna(thresh = len(df_all) - 30, axis = 1)
        df_all.dropna(inplace = True)
        for target in targets:
            print(f'--------- {target} ---------')
            # separate the features and target variable
            X = df_all.drop(targets + ['date', 'rx'], axis=1)
            y = df_all[target]



            # create a SelectKBest object with f_regression scoring
            selector = SelectKBest(mutual_info_regression, k=k)

            # fit the selector to the data and transform the features
            X_selected = selector.fit_transform(X, y)

            # get the indices of the selected features
            selected_indices = selector.get_support(indices=True)

            # get the names of the selected features
            selected_features = X.columns[selected_indices]

            print(list(selected_features))

def merge_optical_feats_and_scans():


    df = pd.read_csv('./Data/processed_optical/features_optical.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.date < pd.Timestamp('2022-09-17 17:30:00')]

    df2 = pd.read_csv('./Data/processed_optical/features_optical_change.csv')
    df2['date'] = pd.to_datetime(df2['date'])
    df2 = df2[df2.date < pd.Timestamp('2022-09-17 17:30:00')]

    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    
    df2.dropna(inplace = True)
    df2.drop_duplicates(inplace = True)
    

    df_optical = pd.read_csv('./Data/raw_nflash230.csv')
    df_optical['date'] = pd.to_datetime(df_optical['date'])
    df_optical = df_optical[df_optical.date < pd.Timestamp('2022-09-17 17:30:00')]
    df_optical = df_optical[['date','COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']]
    df_optical[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']] = np.deg2rad(df_optical[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])

    df_optical.dropna(inplace = True)
    df_optical.drop_duplicates(subset = ['date'], inplace = True)
    
    #inner join on date
    df_merged = pd.merge(df, df_optical, on = 'date', how = 'inner')
    df_merged = pd.merge(df_merged, df2, on = 'date', how = 'inner')

    df_merged['ANAZ'] = np.sin(df_merged['COMMANDAZ']) * np.tan(df_merged['COMMANDEL'])
    df_merged['AWAZ'] = np.cos(df_merged['COMMANDAZ']) * np.tan(df_merged['COMMANDEL'])
    df_merged['NPAE'] = np.tan(df_merged['COMMANDEL'])
    df_merged['CA']   = 1 / np.cos(df_merged['COMMANDEL'])
    df_merged['ANEL'] = np.cos(df_merged['COMMANDAZ'])
    df_merged['AWEL'] = np.sin(df_merged['COMMANDAZ'])

    df_merged['HASA'] =  np.sin(    df_merged.COMMANDAZ) #AWEL
    df_merged['HASA2'] = np.sin(2 * df_merged.COMMANDAZ)
    df_merged['HESA3'] = np.sin(3 * df_merged.COMMANDAZ)
    df_merged['HESA4'] = np.sin(4 * df_merged.COMMANDAZ)
    df_merged['HESA5'] = np.sin(5 * df_merged.COMMANDAZ)
    
    df_merged['HESE'] =  np.sin(    df_merged.COMMANDEL)
    df_merged['HESE2'] = np.sin(2 * df_merged.COMMANDEL)
    df_merged['HESE3'] = np.sin(3 * df_merged.COMMANDEL)
    df_merged['HESE4'] = np.sin(4 * df_merged.COMMANDEL)
    df_merged['HESE5'] = np.sin(5 * df_merged.COMMANDEL)

    # df_merged['HACA'] = np.cos(df_merged.COMMANDAZ) #ANEL
    # df_merged['HACA2'] = np.cos(2 * df_merged.COMMANDAZ)
    # df_merged['HACA3'] = np.cos(3 * df_merged.COMMANDAZ)
    # df_merged['HACA4'] = np.cos(4 * df_merged.COMMANDAZ)
    # df_merged['HACA5'] = np.cos(5 * df_merged.COMMANDAZ)

    df_merged['HECE'] = np.cos(df_merged.COMMANDEL)
    df_merged['HECE2'] = np.cos(2 * df_merged.COMMANDEL)
    df_merged['HECE3'] = np.cos(3 * df_merged.COMMANDEL)
    df_merged['HECE4'] = np.cos(4 * df_merged.COMMANDEL)
    df_merged['HECE5'] = np.cos(5 * df_merged.COMMANDEL)


    # df_merged['HECA3'] = np.cos(3 * df_merged.COMMANDAZ)

    # df_merged['HESA2'] = np.sin(2 * df_merged.COMMANDAZ)

    # df_merged['HECA2'] = np.cos(2 * df_merged.COMMANDAZ)

    # df_merged['HECE'] = np.cos(df_merged.COMMANDEL)



    cos_el = np.cos(df_merged.COMMANDEL)
    df_merged['HSCA'] = np.cos(df_merged.COMMANDAZ) / cos_el
    df_merged['HSCA2'] = np.cos(2 * df_merged.COMMANDAZ) / cos_el
    df_merged['HSCA5'] = np.cos(5 * df_merged.COMMANDAZ) / cos_el

    df_merged['OFFSETAZ'] = (df_merged['ACTUALAZ'] - df_merged['COMMANDAZ']) * np.cos(df_merged['COMMANDEL'])
    df_merged['OFFSETEL'] = df_merged['ACTUALEL'] - df_merged['COMMANDEL']
    df_merged['OFFSETAZ_UNSCALED'] = df_merged['ACTUALAZ'] - df_merged['COMMANDAZ']

    df_merged['OFFSETAZ_PREV'] = df_merged['OFFSETAZ'].shift(1)
    df_merged['OFFSETEL_PREV'] = df_merged['OFFSETEL'].shift(1)
    df_merged['OFFSETAZ_UNSCALED_PREV'] = df_merged['OFFSETAZ_UNSCALED'].shift(1)


    df_merged.dropna(inplace=True)
    
    #Make sure offsetaz, offsetel and offsetaz_unscaled are the 3 last columns
    cols = list(df_merged.columns)
    cols = cols[:-6] + cols[-3:] + cols[-6:-3] 
    df_merged = df_merged[cols]
    

    df_merged.to_csv('./Data/dataset_optical_v2.csv', index = False)

from sklearn.decomposition import PCA
def PCA_on_raw_data():

    # load your dataframe
    df = pd.read_csv('./Data/dataset_optical_v2.csv')

    ignore = ['date', 'OFFSETAZ', 'OFFSETEL', 'OFFSETAZ_UNSCALED', 'OFFSETAZ_PREV', 'OFFSETEL_PREV', 'OFFSETAZ_UNSCALED_PREV']
    df_features = df.loc[: , ~df.columns.isin(ignore)]


    # perform PCA and keep enough components to explain 99% of the variance
    pca = PCA(n_components=0.999)
    pca.fit(df_features)

    # get the loading matrix
    loading_matrix = pd.DataFrame(pca.components_, columns=df_features.columns)
    for i in range(pca.n_components_):
        sorted_loadings = loading_matrix.iloc[i].abs().sort_values(ascending=False)
        top_features = sorted_loadings.index[:3]
        proportions = (sorted_loadings / sorted_loadings.sum())[:3]
        component_df = pd.DataFrame({'Component': 'Component {}'.format(i+1),
                                    'Feature': top_features,
                                    'Proportion': proportions})

        if i == 0:
            components_df = component_df
        else:
            components_df = components_df.append(component_df)
        

    embed()
    components_df.to_csv('./Data/PCA999_component_info_optical_v2.csv', index=False)

    #Transform df
    df_features = pd.DataFrame(pca.transform(df_features), columns=['PCA{}'.format(i) for i in range(pca.n_components_)])
    
    #merge df_features with df[ignore]
    df_merged = pd.concat([df[ignore], df_features], axis=1)
    df_merged.to_csv('./Data/dataset_optical_v2_pca999.csv', index = False)
    


def make_table_with_all_pcorr_features():
    ds1_az = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds1_tp2_k50_uncorr_az')
    ds1_el = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds1_tp2_k50_uncorr_el')
    ds2_az = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds2_tp2_k50_uncorr_az')
    ds2_el = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds2_tp2_k50_uncorr_el')
    ds4_az = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds4_tp2_k50_uncorr_az')
    ds4_el = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds4_tp2_k50_uncorr_el')
    ds6_az = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds6_tp2_k50_uncorr_az')
    ds6_el = pd.read_csv('./FinalResults/Run12/SelectedFeatures/XGB_ds6_tp2_k50_uncorr_el')

    all_features = pd.concat([ds1_az, ds1_el, ds2_az, ds2_el, ds4_az, ds4_el, ds6_az, ds6_el]).drop_duplicates()
    all_features.reset_index(drop = True, inplace = True)
    embed()
    df_all = pd.DataFrame({
        'Features': all_features.features,
        'ds1 az': all_features.features.isin(ds1_az.features),
        'ds1 el': all_features.features.isin(ds1_el.features),
        'ds2 az': all_features.features.isin(ds2_az.features),
        'ds2 el': all_features.features.isin(ds2_el.features),
        'ds4 az': all_features.features.isin(ds4_az.features),
        'ds4 el': all_features.features.isin(ds4_el.features),
        'ds6 az': all_features.features.isin(ds6_az.features),
        'ds6 el': all_features.features.isin(ds6_el.features)
    })

    df_all.replace({True: 'x', False: ''}, inplace = True)
    df_all.sort_values(by='Features', inplace=True)
    df_all.to_latex('./FinalResults/Run12/SelectedFeatures/all_features.tex', index=False, longtable = True)

if __name__ == '__main__':
    #correction_check()
    #check_scan_timestamps()    
    features = ['ACTUALAZ', 'ACTUALEL', 'WINDSPEED', 'WINDDIRECTION', 'TILT2X',
                'TILT2Y', 'TILT2T', 'TILT3X', 'TILT3Y', 'TILT3T','ACTUALVELOCITYAZ',
                'ACTUALVELOCITYEL','REFERENCEAZ','REFERENCEEL','COMMANDAZ','COMMANDEL',
                'DISP_ABS1','DISP_ABS2','DISP_ABS3','DAZ_TILT','DAZ_TILTTEMP','DAZ_TOTAL',
                'DEL_DISP','DEL_SPEM','DEL_TEMP','DEL_TILT','DEL_TILTTEMP','DEL_TOTAL']

    # merge_optical_feats_and_scans()

    # feature_selection_final(5)

    # data_frequency_processed_data('./Data/db_exports/ACTUALAZ.csv')
    # plotScanFeature_v2(['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL'])
    #cluster_bad_scans()
    # plotScanFeature_optical(['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL'])
    # scan_duration_distribution()
    # corr_temp()
    #move_random_imgs()
    # PCA_on_raw_data()
    scan_duration_distribution()
    # make_table_with_all_pcorr_features()
    # plot_df_corr()
    # merge_optical_feats_and_scans()
    # scan_duration_distribution()

    """
    path_df = './Data/processed_v3/ALL_MEDIAN_during_0_lower.csv'
    df = pd.read_csv(path_df)
    embed()
    for pk in [0,1,2,3,4,10,11,12,13]:
        analysis_pipeline(path_integrated_pos, pk)"""

    """    df = pd.read_csv('./Data/tmp2022.csv')
    df2 = pd.read_csv('./Data/PointingTable.csv')

    print(len(df))
    df2 = df2[df2.scan.isin(df.scan.unique())]
    print(len(df))"""
    
    """
    PATH = './Data/processed_v3/all_features_safe.csv'
    for key in range(0,5):
        test = Analysis(PATH, remove_outliers=False, patch_key = key)
        test.plot_correlation()



    analyze_tiltmeter()
    pass
    check_duplicate_datetime()
    pass
    df = pd.read_csv('./Data/PointingTable2.csv')

    print(df.describe())
    dates = df["obs_date"]
    df2 = df[dates.isin(dates[dates.duplicated()])]
    print(df2['rx'].value_counts())"""



    #Analysis with new meged csv
    # merge_features(path_median, path_median_wind, path_median_sun, all_vals=True)
    #scan_locations3D()  
    #embed()
    #analysis = Analysis(path_df = './Data/merged_features_all.csv', patch=fp_lines[6])
    #analysis.scan_locations()
    #analysis.plot_correlation_sun()
    #read pointing table into df


def integrate(df, value):
    #given dataframe with datetime and value, integrate over the change in value over time
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df['diff'] = df[value].diff()
    df['diff'] = df['diff'].fillna(0)
    df['integrated'] = df['diff'].cumsum()
    df['integrated'] = df['integrated'].fillna(0)
    df = df.reset_index()

    return df


"""
Average frequency of data points per minute

date                44.640166
ACTUALAZ             5.928120
ACTUALEL             5.928120
DEWPOINT             5.177097
HUMIDITY             5.177097
PRESSURE             5.177024
TEMPERATURE          5.177097
WINDDIRECTION        5.177082
WINDSPEED            5.177061
MODE                 1.831023
TILT1X              11.865223
TILT1Y              11.865223
TILT1T              11.865223
TILT2X              11.865223
TILT2Y              11.865223
TILT2T              11.865223
TILT3X              11.865223
TILT3Y              11.865223
TILT3T              11.865223
DAZ_DISP            11.693869
DAZ_SPEM             7.488797
DAZ_TEMP             1.831023
DAZ_TILT            11.865223
DAZ_TILTTEMP        11.733970
DAZ_TOTAL           11.693883
DEL_DISP            11.693869
DEL_SPEM             7.488797
DEL_TEMP             1.831023
DEL_TILT            11.865223
DEL_TILTTEMP        11.733970
DEL_TOTAL           11.693869
POSITIONX            5.862248
POSITIONY            5.862233
POSITIONZ            5.862233
ROTATIONX            5.848545
ROTATIONY            5.848545
ROTATIONZ            5.848545
TEMP1                5.554741
TEMP2                5.554741
TEMP3                5.554741
TEMP4                5.554741
TEMP5                5.554741
TEMP6                5.554741
TEMP26               1.977666
TEMP27               1.977666
TEMP28               1.977666
ACTUALVELOCITYAZ     0.004200
ACTUALVELOCITYEL     0.004200
REFERENCEAZ          0.004200
REFERENCEEL          0.004200
COMMANDAZ            0.004200
COMMANDEL            0.004200
DISP_ABS1           11.733970
DISP_ABS2           11.733970
DISP_ABS3           11.733970

"""