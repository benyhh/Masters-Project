from concurrent.futures import process
from turtle import color
import functions
import importlib
importlib.reload(functions)
from functions import *
from functions import random_seed
import multiprocessing as mp

random.seed(random_seed)
np.random.seed(random_seed)

path_unprocessed_df = './Data/processed' + 'all_scans_15_3.csv'
path_processed_sun  = './Data/processed' + 'sun_features_15_3.csv'
path_processed_wind = './Data/processed' + 'wind_features_15_3.csv'
PATH_RAW            = './Data/raw/'      + 'raw_data_15_10.csv'

# ignore_cols = ['DAZ_DISP', 'DAZ_SPEM', 'DAZ_TEMP', 'DAZ_TILT',
#        'DAZ_TILTTEMP', 'DAZ_TOTAL', 'DEL_DISP', 'DEL_SPEM', 'DEL_TEMP',
#        'DEL_TILT', 'DEL_TILTTEMP', 'DEL_TOTAL', "MODE"]


class DataProcessing():
    """
    Class for transforming data. Reads given data frame and performs transformations on it.

    Takes in parameters for different transformations and saves the new df
    """
    def __init__(self, path_scans = None):

        #print(self.df.describe())

        #Load pointing table, it has time of scans
        if path_scans is None:
            df_scans             = pd.read_csv('./Data/tmp2022_clean_v2.csv')
            df_scans['obs_date'] = pd.to_datetime(df_scans['obs_date'])
            df_scans = df_scans[df_scans['obs_date'] < pd.Timestamp('2022-09-18')]
            self.df_scans = df_scans

        else:
            df_scans = pd.read_csv(path_scans)
            df_scans['date'] = pd.to_datetime(df_scans['date'])
            self.df_scans = df_scans
    
    
        df_durations = pd.read_csv('./Data/df_scanDuration.csv')
        df_durations['date'] = pd.to_datetime(df_durations['date'])
        df_durations['start_observing'] = pd.to_datetime(df_durations['start_observing'])
        df_durations['end_observing']   = pd.to_datetime(df_durations['end_observing'])
        self.df_durations = df_durations 

        df_frequencies = pd.read_csv('./Data/data_frequency_last.csv')
        df_frequencies = df_frequencies.set_index('Feature')
        df_frequencies = df_frequencies['min_per_point']
        self.dict_frequency = df_frequencies.to_dict()
        print(self.dict_frequency)
    def get_last(self, df):
        if df.empty:
            return np.nan
        else:
            return df.iloc[-1]

    def get_median(self, df):
        if len(df) % 2 == 0:
            return df.iloc[:-1].median()
        else:
            return df.median().item()

    def get_variance(self, df):
        return df.var()

    def get_mean(self, df):
        return df.mean()

    def get_sum_dabs(self, df):
        if len(df) < 2:
            return np.nan
        else:
            return df.diff().abs().sum(min_count = 1)

    def get_mean_dabs(self, df):
        if len(df) < 2:
            return np.nan
        else:
            return df.diff().abs().mean()

    def get_change(self, df):
        if len(df) < 2:
            return np.nan
        else:
            return df.iloc[-1] - df.iloc[0]
    
    def get_max(self, df):
        return df.max()
    
    def get_max_change_in_interval(self, df, interval):
        max_change = df.diff(interval).max()
        min_change = df.diff(interval).min()
        return min_change if abs(min_change) > max_change else max_change

    def get_change_in_median(self, df, interval):
        df_start = df.iloc[:interval]
        df_end   = df.iloc[-interval:]
        return self.get_median(df_end) - self.get_median(df_start)

    def integrate_positive_change(self, df):
        try:
            return df.diff(1).clip(lower=0).sum(min_count = 1)
        except:
            embed(header = 'integrate_positive_change')

    def integrate_negative_change(self, df):
        return df.diff(1).clip(upper=0).sum(min_count = 1)

    def integrate_absolute_change(self, df):
        return df.diff(1).abs().sum(min_count = 1)


    def raw_data_processing(self):
        """
        Transforming raw data. Used for all unproccessed dfs from the database.
        """

        df = self.df.copy()

        df.loc[:, "ACTUALAZ"] = np.rad2deg(df["ACTUALAZ"])
        df.loc[:, "ACTUALEL"] = np.rad2deg(df["ACTUALEL"])


        #use self.get_sun_position to get sun position for all datetimes in column date, and insert az and el into df
        df.loc[:, "SUN_AZ"], df.loc[:, "SUN_EL"] = self.get_sun_position(df['date'])
        df.loc[ df['SUN_AZ'] > 180, "SUN_AZ"] -= 360

        #insert column that is the difference between azimuth angle and sun azimuth and keep the interval [-180, 180]
        df.loc[:, "SUN_AZ_D"] = df["ACTUALAZ"] - df["SUN_AZ"]
        df.loc[df["SUN_AZ_D"] > 180, "SUN_AZ_D"] -= 360
        df.loc[df["SUN_AZ_D"] < -180, "SUN_AZ_D"] += 360
        df.loc[:, "ANGLE_TO_SUN"] = calcSmallestAngle(df['SUN_AZ'], df['SUN_EL'], df['ACTUALAZ'], df['ACTUALEL'])

        #set all values of sun az and el, and sunaz_d to 0 if sun is below horizon
        df.loc[df['SUN_EL'] < 0, 'SUN_AZ']   = 0
        df.loc[df['SUN_EL'] < 0, 'SUN_EL']   = 0
        df.loc[df['SUN_EL'] < 0, 'SUN_AZ_D'] = 0
        #insert column that is the difference between azimuth angle and wind direction and keep the interval [-180, 180]
        #Azimuth and wind data are not on the exact same timestamp. Make it so that win is moved to the closest azimuth timestamo
        df.loc[df['WINDDIRECTION'] > 180, "WINDDIRECTION"] -= 360 
        df.loc[:, "WINDDIRECTION"] = df["WINDDIRECTION"].interpolate(method='linear')

        df.loc[:, "WINDDIRECTION_D"] = df["ACTUALAZ"] - df["WINDDIRECTION"]
        df.loc[df["WINDDIRECTION_D"] > 180, "WINDDIRECTION_D"] -= 360
        df.loc[df["WINDDIRECTION_D"] < -180, "WINDDIRECTION_D"] += 360
        
        df.to_csv('./Data/processed_v2/processed_data_15_10.csv', index = False)

    def get_scan_values_optical(self, feature_params = {}):

        actions = {
                'LAST': self.get_last,
                'MEDIAN': self.get_median,
                'VARIANCE': self.get_variance, 
                'MEAN': self.get_mean,
                'SUM_DABS': self.get_sum_dabs,
                'MEAN_DABS': self.get_mean_dabs,
                'CHANGE': self.get_change,
                'MAX': self.get_max,
                'MAX_CHANGE_IN_INTERVAL': self.get_max_change_in_interval,
                'CHANGE_IN_MEDIAN': self.get_change_in_median,
                'INTEGRATE_POSITIVE_CHANGE': self.integrate_positive_change,
                'INTEGRATE_NEGATIVE_CHANGE': self.integrate_negative_change,
                'INTEGRATE_ABSOLUTE_CHANGE': self.integrate_absolute_change,
                }

        df_scans     = self.df_scans
        #Date previous scan
        df_scans.sort_values(by='date', inplace=True)
        df_scans['timeLastCorrection'] = df_scans['date'].shift(1)
        df_scans.dropna(inplace=True)

        PATH_RAW_DATA = './Data/db_exports/'
        PATH_OUTPUT = './Data/processed_optical/features_optical_change.csv'
        df_output = pd.DataFrame(data = {'date': df_scans['date']})


        for key, params in feature_params.items():
            
            print(f'Getting features for key {key}')

            if 'period' in params:
                
                for i,feature in enumerate(params['columns']):

                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    
                    timeBeforeScan = pd.Timedelta(f'{params["period"]}m')
                    
                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):
                        intervalStart = rowScan.date - timeBeforeScan
                        intervalEnd   = rowScan.date

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature])
                        feature_values.append(feature_value)
                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values
                    print('Writing current output data to file')
                    df_output.to_csv(PATH_OUTPUT, index = False)     
                
            elif 'around' in params:
                
                for i,feature in enumerate(params['columns']):

                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    
                    timeAroundScan = pd.Timedelta(f'{params["around"]}m')
                    
                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):
                        intervalStart = rowScan.date - timeAroundScan
                        intervalEnd   = rowScan.date + timeAroundScan

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature])
                        feature_values.append(feature_value)
                    
                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values
                    print('Writing current output data to file')
                    df_output.to_csv(PATH_OUTPUT, index = False)    

            elif 'interval' in params:
                """
                Extracts features from intervals since last scan.
                """

                for i,feature in enumerate(params['columns']):
                    
                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    interval = int(params['interval'] / self.dict_frequency[feature])   #data_freq is minutes/points, and interval is minutes

                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    


                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):

                        intervalStart = df_scans[df_scans.date == rowScan.date].timeLastCorrection.values[0] 
                        intervalEnd   = df_scans[df_scans.date == rowScan.date].date.values[0] 

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature], interval)
                        feature_values.append(feature_value)

                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values
                    print('Writing current output data to file')
                    df_output.to_csv(PATH_OUTPUT, index = False)      
                
            else:
                for i,feature in enumerate(params['columns']):

                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    

                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):

                        intervalStart = df_scans[df_scans.date == rowScan.date].timeLastCorrection.values[0]
                        intervalEnd   = df_scans[df_scans.date == rowScan.date].date.values[0] 

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature])

                        feature_values.append(feature_value)

                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values
                    print('Writing current output data to file')
                    df_output.to_csv(PATH_OUTPUT, index = False)            




    def get_scan_values_v3(self, feature_params = {}, PATH_OUTPUT = './Data/scanvals_v3.csv'):

        actions = {
                'LAST': self.get_last,
                'MEDIAN': self.get_median, 
                'VARIANCE': self.get_variance, 
                'MEAN': self.get_mean,
                'SUM_DABS': self.get_sum_dabs,
                'MEAN_DABS': self.get_mean_dabs,
                'CHANGE': self.get_change,
                'MAX': self.get_max,
                'MAX_CHANGE_IN_INTERVAL': self.get_max_change_in_interval,
                'CHANGE_IN_MEDIAN': self.get_change_in_median,
                'INTEGRATE_POSITIVE_CHANGE': self.integrate_positive_change,
                'INTEGRATE_NEGATIVE_CHANGE': self.integrate_negative_change,
                'INTEGRATE_ABSOLUTE_CHANGE': self.integrate_absolute_change,
                }

        df_scans     = self.df_scans
        df_durations = self.df_durations


        PATH_RAW_DATA = './Data/db_exports/'
        df_output = pd.DataFrame(data = {'date': df_scans['obs_date']})
        print('Path output', PATH_OUTPUT)
        
        for key, params in feature_params.items():
            
            print(f'Getting features for key {key}')

            if 'during' in params:
                
                for i,feature in enumerate(params['columns']):

                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    
                    
                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):

                        intervalStart = df_durations[df_durations.date == rowScan.obs_date].start_observing.values[0]
                        intervalEnd   = df_durations[df_durations.date == rowScan.obs_date].end_observing.values[0]

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature])
                        feature_values.append(feature_value)
                    
                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values
                


            elif 'period' in params:
                """
                Features from a period in minutes before the scan
                """
                timeBeforeScan = pd.Timedelta(f'{params["period"]}m')

                for i,feature in enumerate(params['columns']):
                    
                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')

                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    


                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):

                        intervalStart = df_durations[df_durations.date == rowScan.obs_date].start_observing.values[0] - timeBeforeScan
                        intervalEnd   = df_durations[df_durations.date == rowScan.obs_date].start_observing.values[0]

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature])

                        feature_values.append(feature_value)
                    
                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values



            elif 'interval' in params:
                """
                Extracts features from intervals since last scan.
                """

                for i,feature in enumerate(params['columns']):
                    
                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    interval = int(params['interval'] / self.dict_frequency[feature])   #data_freq is minutes/points, and interval is minutes

                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    


                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):
                        
                        timeLastCorrection = df_scans[df_scans.obs_date == rowScan.obs_date].timeLastCorrection.values[0] 

                        intervalStart = df_durations[df_durations.date == timeLastCorrection].end_observing.values[0]
                        intervalEnd   = df_durations[df_durations.date == rowScan.obs_date].start_observing.values[0]

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature], interval)
                        feature_values.append(feature_value)

                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values

            else:
                for i,feature in enumerate(params['columns']):

                    print(f'Feature: {feature} ({i+1}/{len(params["columns"])})')
                    
                    feature_path = os.path.join(PATH_RAW_DATA, feature + '.csv')
                    df_feature = pd.read_csv(feature_path)
                    df_feature['date'] = pd.to_datetime(df_feature['date'])
                    

                    feature_values = []

                    for rowScan in tqdm(df_scans.itertuples()):
                        
                        timeLastCorrection = df_scans[df_scans.obs_date == rowScan.obs_date].timeLastCorrection.values[0] 

                        intervalStart = df_durations[df_durations.date == timeLastCorrection].end_observing.values[0]
                        intervalEnd   = df_durations[df_durations.date == rowScan.obs_date].start_observing.values[0]

                        df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                        feature_value = actions[params['operation']](df_feature_interval[feature])

                        feature_values.append(feature_value)


                    column_name = feature + '_' + params['suffix']
                    df_output[column_name] = feature_values           

            print('Writing current output data to file')
            df_output.to_csv(PATH_OUTPUT, index = False)
          


    def get_scan_values_v0(self, timeBeforeScan = 0, duringScan=True, operation = 'MEDIAN', name = 'SCANS', selectedColumns = None, feature_parameters = {}):

        actions = {
                'LAST': self.get_last,
                'MEDIAN': self.get_median, 
                'VAR': self.get_variance, 
                'MEAN': self.get_mean,
                'SUM_DABS': self.get_sum_dabs,
                'MEAN_DABS': self.get_mean_dabs,
                'CHANGE': self.get_change}

        df_durations = pd.read_csv('./Data/df_scanDuration.csv')
        df_durations['date'] = pd.to_datetime(df_durations['date'])
        df_durations['start_observing'] = pd.to_datetime(df_durations['start_observing'])
        df_durations['end_observing']   = pd.to_datetime(df_durations['end_observing'])

        if duringScan:
            beforeOrDuring = 'during'
        else:
            beforeOrDuring = 'before'

        fn = f'./Data/processed_v2/{name}_{operation}_{beforeOrDuring}_{timeBeforeScan}_{estimateType}.csv'

        timeBeforeScanTimedelta = pd.Timedelta(minutes = timeBeforeScan)
        
        PATH_RAW_DATA = './Data/db_exports/'

        #make empty df with a date column

        if selectedColumns is None:
            data_files = os.listdir(PATH_RAW_DATA)
            selectedColumns = [f.split('.')[0] for f in data_files]

        df_features = pd.DataFrame(data = {'date': self.df_scans.obs_date})    
        
        for feature in selectedColumns:

            feature_path = os.path.join(PATH_RAW_DATA, _col + '.csv')
            df_feature = pd.read_csv(feature_path)
            df_feature['date'] = pd.to_datetime(df_feature['date'])
            
            feature_values = []
            for rowScan in self.df_scans.itertuples():
                

                start_observing = df_durations[df_durations.date == rowScan.obs_date].start_observing.values[0]
                end_observing   = df_durations[df_durations.date == rowScan.obs_date].end_observing.values[0]

                if duringScan:
                    intervalStart = start_observing
                    intervalEnd   = end_observing
                elif not duringScan:
                    intervalStart = start_observing - timeBeforeScanTimedelta
                    intervalEnd   = start_observing

                df_feature_interval = df_feature[(df_feature.date >= intervalStart) & (df_feature.date <= intervalEnd)]
                feature_value = actions[operation](df_feature_interval[feature])
                feature_values.append(feature_value)
            
            df_features[feature] = feature_values

    def get_scan_values_v2(self, timeBeforeScan = 0, duringScan=True, operation = 'MEDIAN', useAllScans = True, name = 'SCANS',estimateType = 'lower', selectedColumns = None):
        """
        Extracts the values closest to the pointing scan and writes to csv file.
        Parameters
        dt0_: how long after the pointing scan it looks
        dt1_: how long before the pointing scan it looks
        operation: what values to find, i.e. last, median, variance, etc. 
        """

        df_scanDuration = pd.read_csv('./Data/scan_durations2022.csv')
        df_scanDuration['date'] = pd.to_datetime(df_scanDuration['date'])

        actions = {
                'LAST': self.get_last,
                'MEDIAN': self.get_median, 
                'VAR': self.get_variance, 
                'MEAN': self.get_mean,
                'SUM_DABS': self.get_sum_dabs,
                'MEAN_DABS': self.get_mean_dabs,
                'CHANGE': self.get_change}
        
        if duringScan:
            beforeOrDuring = 'during'
        else:
            beforeOrDuring = 'before'

        if useAllScans:
            fn = f'./Data/processed_v2/{name}_{operation}_{beforeOrDuring}_{timeBeforeScan}_{estimateType}_allScans.csv'
        else:
            fn = f'./Data/processed_v2/{name}_{operation}_{beforeOrDuring}_{timeBeforeScan}_{estimateType}_onlyFlag.csv'


        print(f"Starting to extract values {beforeOrDuring} scan | timeBeforeScan = {timeBeforeScan} | operation = {operation}")
        timeBeforeScanTimedelta = pd.Timedelta(minutes = timeBeforeScan)

        ignore_cols = ['MODE', 'ROTATIONZ', 'ACTUALVELOCITYAZ','ACTUALVELOCITYEL','REFERENCEAZ','REFERENCEEL','COMMANDAZ','COMMANDEL']
        df_rawData = self.df.loc[:, ~self.df.columns.isin(ignore_cols)]
        
        if selectedColumns is not None:
            df_rawData = df_rawData.loc[: , df_rawData.columns.isin(['date'] + selectedColumns)]
    
        df_values = pd.DataFrame()

        count = 0
        for rowScan in tqdm(self.df_scans.itertuples()):
            
            scanFlagsCurrentScan = df_scanDuration.loc[(df_scanDuration['date'] == rowScan.obs_date) & (df_scanDuration['rx'] == rowScan.rx)]

            if not scanFlagsCurrentScan.empty:
                start_observing, end_observing = scanFlagsCurrentScan.start.values[0], scanFlagsCurrentScan.end.values[0]
            else:
                start_observing = None


            if start_observing is None and useAllScans is True:
                
                start_observing = rowScan.obs_date + self.get_scan_diff_mean(rowScan.rx, estimateType=estimateType)

                if duringScan:
                    intervalStart = start_observing - timeBeforeScanTimedelta
                    intervalEnd   = start_observing + self.get_scan_duration_mean(rowScan.rx, estimateType=estimateType)
                
                else:
                    intervalStart = start_observing - timeBeforeScanTimedelta
                    intervalEnd   = start_observing


            else:
                count += 1

            _row = {'date': rowScan.obs_date}
            # Filtering data from intervalStart and intervalEnd
            df_scan = df_rawData.loc[ (df_rawData.date > intervalStart) & (df_rawData.date < intervalEnd), :]
            #print(df_t.describe())
            for _col in df_rawData.columns:
                
                if _col == 'date' or _col == 'scan':
                    continue
                
                df_col     = df_scan.loc[df_scan[_col].notna(), _col]
                try:
                    _row[_col] = actions[operation](df_col)
                except:
                    _row[_col] = np.nan
                
            _row = pd.DataFrame(_row, index = [0])
            df_values = pd.concat([df_values, _row], ignore_index=True)

        print(f'Found {count}/{len(self.time_scans)} scans with exact start/end')

        
        df_values.to_csv(fn, index = False)
        
        return

    def get_scan_values(self, dt0_ = 0, dt1_ = 0, operation = 'median', all_vals = True, name = 'scans'):
        """
        Extracts the values closest to the pointing scan and writes to csv file.
        Parameters
        dt0_: how long after the pointing scan it looks
        dt1_: how long before the pointing scan it looks
        operation: what values to find, i.e. last, median, variance, etc. 
        """

        actions = {
                'last': self.get_last,
                'median': self.get_median, 
                'variance': self.get_variance, 
                'mean': self.get_mean,
                'sum_dabs': self.get_sum_dabs,
                'mean_dabs': self.get_mean_dabs}
        
        dt0 = pd.Timedelta(minutes = dt0_)
        dt1 = pd.Timedelta(minutes = dt1_)

        time_offset   = pd.Timedelta(seconds = 53)
        scan_duration = pd.Timedelta(seconds = 90)
        ignore_cols = ['MODE', 'ROTATIONZ', 'ACTUALVELOCITYAZ','ACTUALVELOCITYEL','REFERENCEAZ','REFERENCEEL','COMMANDAZ','COMMANDEL']
        df_filtered = self.df.loc[:, ~self.df.columns.isin(ignore_cols)]
        df_values = pd.DataFrame()

        print(f"Starting to extract values | dt0 = {dt0_} | dt1 = {dt1_} | operation = {operation}")
        count = 0
        for t in tqdm(self.time_scans):
            
            start_observing, end_observing = get_scan_flags(t)

            if start_observing is None:
                if all_vals:
                    start_observing = t + time_offset
                    end_observing   = start_observing + scan_duration
                else:
                    continue
            else:
                count += 1

            _row = {'date': t}
            # Filtering data from dt0 before to dt1 after scan.
            df_t = df_filtered.loc[ (df_filtered.date > start_observing-dt0) & (df_filtered.date < end_observing + dt1), :]
            #print(df_t.describe())
            for _col in df_filtered.columns:
                
                if _col == 'date' or _col == 'scan':
                    continue
                
                df_col     = df_t.loc[df_t[_col].notna(), _col]
                try:
                    _row[_col] = actions[operation](df_col)
                except:
                    _row[_col] = np.nan
                
            _row = pd.DataFrame(_row, index = [0])
            df_values = pd.concat([df_values, _row], ignore_index=True)

        print(f'Found {count}/{len(self.time_scans)} scans with exact start/end')

        if all_vals:
            df_values.to_csv(f'./Data/{name}_{operation}_{dt0_}_{dt1_}_all.csv', index = False)
        else:
            df_values.to_csv(f'./Data/{name}_{operation}_{dt0_}_{dt1_}.csv', index = False)
        
        return
        



    def get_sun_position(self, date_col):
        apex = ephem.Observer()
        apex.lat  = '-23:00:20.8'
        apex.long = '-67:45:33.0'
        apex.elev =   5105

        date_str = date_col.dt.strftime('%Y/%m/%d %H:%M')
        
        sun = ephem.Sun()

        Az_sun = np.zeros(len(date_str))
        El_sun = np.zeros(len(date_str))
        print('Getting sun position')
        for i in range(len(date_str)):
            apex.date = date_str[i]
            sun.compute(apex)
            Az_sun[i] = np.degrees(sun.az)
            El_sun[i] = np.degrees(sun.alt)

        return Az_sun, El_sun

    
    def get_sun_features(self):
        """
        Extracts the sun features and writes to csv file.
        """


        Az_sun, El_sun = self.get_sun_position(self.df.loc[ :, 'date'])
        dfsun = pd.DataFrame({'date': self.df['date'], 'Az_sun': Az_sun, 'El_sun': El_sun})

        dfsun = dfsun.drop_duplicates()
        dfsun.loc[ dfsun['Az_sun'] > 180, 'Az_sun' ] -= 360
        dfsun = dfsun.merge(self.df.loc[ :, ['date', 'ACTUALAZ', 'ACTUALEL']], on = 'date', how = 'right')
        
        dfsun.loc[dfsun['ACTUALAZ'] >  180, 'ACTUALAZ'] -= 360
        dfsun.loc[dfsun['ACTUALAZ'] < -180, 'ACTUALAZ'] += 360

        dfsun.insert(0, 'SunAzDiff'      , dfsun['ACTUALAZ'] - dfsun['Az_sun'])
        dfsun.insert(0, 'SunElDiff'      , dfsun['ACTUALEL'] - dfsun['El_sun'])
        dfsun.insert(0, 'SunAboveHorizon', np.where(dfsun['El_sun'].values > 0, 1, 0))
        dfsun.insert(0, 'SunAngleDiff'   , calcSmallestAngle(dfsun['Az_sun'], dfsun['El_sun'], dfsun['ACTUALAZ'], dfsun['ACTUALEL']))


        dfsun.loc[ dfsun['SunAzDiff'] >  180, 'SunAzDiff' ] -= 360
        dfsun.loc[ dfsun['SunAzDiff'] < -180, 'SunAzDiff' ] += 360
        

        print(dfsun.describe())
        print(dfsun.dropna())
        dfsun = dfsun.dropna()

        dfsun.loc[:, ~dfsun.columns.isin(['ACTUALAZ', 'ACTUALEL']) ].to_csv('./Data/sun_features_15_3.csv', index = False)

        return
    
    def get_wind_features(self):
        """
        Extracts wind features and writes to csv file.
        """

        dfwind = self.df.loc[ :, ['date', 'WINDSPEED', 'WINDDIRECTION']].drop_duplicates().dropna()
        dfaz   = self.df.loc[ :, ['date', 'ACTUALAZ']                  ].drop_duplicates().dropna()
        
        dfaz = dfaz.set_index('date').reindex(dfwind.set_index('date').index, method='nearest').reset_index()
        
        dfwind = dfwind.merge(dfaz, left_on = 'date', right_on= 'date', how='inner')


        dfwind.loc[ dfwind['ACTUALAZ'] >  180, 'ACTUALAZ'] -= 360
        dfwind.loc[ dfwind['ACTUALAZ'] < -180, 'ACTUALAZ'] += 360

        dfwind.insert(0, 'WINDDIR DIFF', dfwind['ACTUALAZ'] - dfwind['WINDDIRECTION'])

        dfwind.loc[ dfwind['WINDDIR DIFF'] >  180, 'WINDDIR DIFF' ] -= 360
        dfwind.loc[ dfwind['WINDDIR DIFF'] < -180, 'WINDDIR DIFF' ] += 360

        dfwind.loc[:, ~dfwind.columns.isin(['ACTUALAZ', 'WINDDIRECTION']) ].to_csv('./Data/wind_features_15_3.csv', index = False)
        print(dfwind.describe())

        return

    def get_scan_values_sun(self, operation = 'median', all_vals = True):
        """
        Extracts the values closest to the pointing scan and writes to csv file.
        Parameters
        dt0_: how long after the pointing scan it looks
        dt1_: how long before the pointing scan it looks
        operation: what values to find, i.e. last, median, variance, etc. 
        """
        actions = {
                'last': self.get_last,
                'median': self.get_median, 
                'variance': self.get_variance, 
                'mean': self.get_mean,
                'sum_dabs': self.get_sum_dabs}

        df_filtered = self.df.loc[:, ~self.df.columns.isin(ignore_cols)]
        df_values = pd.DataFrame()

        dt0 = pd.Timedelta(minutes = 0)
        dt1 = pd.Timedelta(minutes = 0)
        
        time_offset   = pd.Timedelta(seconds = 53)
        scan_duration = pd.Timedelta(seconds = 90)

        count = 0
        for t in tqdm(self.time_scans):
            
            start_observing, end_observing = get_scan_flags(t)

            if start_observing is None:
                if all_vals:
                    start_observing = t + time_offset
                    end_observing   = start_observing + scan_duration
                else:
                    continue
            else:
                count += 1

            _row = {'date': t}
            # Filtering data from dt0 before to dt1 after scan.

            df_t = df_filtered.loc[ (df_filtered.date > start_observing - dt0) & (df_filtered.date < end_observing + dt1), :]
            #print(df_t.describe())

            for _col in df_filtered.columns:
                
                if _col == 'date' or _col == 'scan':
                    continue
                
                df_col     = df_t.loc[df_t[_col].notna(), _col]
                try:
                    _row[_col] = actions[operation](df_col)
                except:
                    _row[_col] = np.nan
                
                if _col == 'SunAngleDiff':
                    df_t2 = df_filtered.loc[ (df_filtered.date > start_observing - pd.Timedelta(minutes = 15)) & (df_filtered.date < end_observing), :]
                    try:
                        _row['SunAngleDiff_15'] = actions[operation](df_t2['SunAngleDiff'])
                    except:
                        _row['SunAngleDiff_15'] = np.nan
            
            _row = pd.DataFrame(_row, index = [0])
            df_values = pd.concat([df_values, _row], ignore_index=True)


        print(f'Found {count}/{len(self.time_scans)} scans')
        if all_vals:
            df_values.to_csv(f'./Data/scans_sun_{operation}_{0}_{0}_all.csv', index = False)
        else:
            df_values.to_csv(f'./Data/scans_sun_{operation}_{0}_{0}.csv', index = False)
        return

    def get_scan_values_wind(self, operation = 'median', all_vals = True):
        """
        Extracts the values closest to the pointing scan and writes to csv file.
        Parameters
        dt0_: how long after the pointing scan it looks
        dt1_: how long before the pointing scan it looks
        operation: what values to find, i.e. last, median, variance, etc. 
        """
        actions = {
                'last': self.get_last,
                'median': self.get_median, 
                'variance': self.get_variance,
                'mean': self.get_mean,
                'sum_dabs': self.get_sum_dabs}

        df_filtered = self.df.loc[:, ~self.df.columns.isin(ignore_cols)]
        df_values = pd.DataFrame()

        dt0_num  = 2
        dt1_num = 0

        dt0 = pd.Timedelta(minutes = dt0_num)
        dt1 = pd.Timedelta(minutes = dt1_num)

        time_offset   = pd.Timedelta(seconds = 53)
        scan_duration = pd.Timedelta(seconds = 90)
        
        count = 0
        for t in tqdm(self.time_scans):
            
            start_observing, end_observing = get_scan_flags(t)

            if start_observing is None:
                if all_vals:
                    start_observing = t + time_offset
                    end_observing   = start_observing + scan_duration
                else:
                    continue
            else:
                count += 1

            _row = {'date': t}
            # Filtering data from dt0 before to dt1 after scan.

            df_t = df_filtered.loc[ (df_filtered.date > start_observing - dt0) & (df_filtered.date < end_observing + dt1), :]
            #print(df_t.describe())

            for _col in df_filtered.columns:
                
                if _col == 'date' or _col == 'scan':
                    continue
                
                df_col     = df_t.loc[df_t[_col].notna(), _col]
                try:
                    _row[_col] = actions[operation](df_col)
                except:
                    _row[_col] = np.nan
                
                if _col == 'WINDSPEED':
                    try:
                        _row['TURBULENCE'] = np.std(df_t[_col].values)
                    except:
                        _row['TURBULENCE'] = np.nan

            _row = pd.DataFrame(_row, index = [0])
            df_values = pd.concat([df_values, _row], ignore_index=True)


        print(f'Found {count}/{len(self.time_scans)} scans')
        if all_vals:
            df_values.to_csv(f'./Data/scans_wind_{operation}_{dt0_num}_{dt1_num}_all.csv', index = False)
        else:
            df_values.to_csv(f'./Data/scans_wind_{operation}_{dt0_num}_{dt1_num}.csv', index = False)
        return
   
    def get_scan_duration_mean(self, rx, estimateType='lower'):
        #create dictionry with duration with rx as key
        lower = {
            'CHAMP690': 47.181818,
            'HOLO': 26.809524,
            'LASMA345': 66.668874,
            'NFLASH230': 63.857527,
            'NFLASH460': 63.730469,
            'SEPIA180': 63.736111,
            'SEPIA345': 69.669355,
            'SEPIA660': 59.781726,
            'ZEUS2': 76.333333
        }

        upper = {
            'CHAMP690': 251.666667,
            'LASMA345': 190.820408,
            'NFLASH230': 130.408403,
            'NFLASH460': 190.395745,
            'SEPIA180': 273.526316,
            'SEPIA345': 188.137037,
            'SEPIA660': 193.735537,
            'HOLO': 26.809524,
            'ZEUS2': 76.333333
        
        }

        middle = {
            'CHAMP690': 91.000000,
            'HOLO': 26.809524,
            'LASMA345': 143.479798,
            'NFLASH230': 104.801653,
            'NFLASH460': 124.354379,
            'SEPIA180': 192.317204,
            'SEPIA345': 150.724051,
            'SEPIA660': 133.624146,
            'ZEUS2': 76.333333
        }


        if estimateType == 'lower':
            return pd.Timedelta(seconds = lower[rx])
        
        elif estimateType == 'upper':
            return pd.Timedelta(seconds = upper[rx])
        
        elif estimateType == 'middle':
            return pd.Timedelta(seconds = middle[rx])
        
        else:
            print('Wrong estimateType. Choose between lower, upper and middle')
            return None



    def get_scan_diff_mean(self, rx, estimateType='lower'):
        lower = {
            'CHAMP690': 58.545455,
            'HOLO': 41.952381,
            'LASMA345': 56.119205,
            'NFLASH230': 47.723118,
            'NFLASH460': 52.125000,
            'SEPIA180': 50.208333,
            'SEPIA345': 52.241935,
            'SEPIA660': 57.878173,
            'ZEUS2': 51.393939
        }

        upper = {
            'CHAMP690': 57.333333,
            'LASMA345': 54.473469,
            'NFLASH230': 55.349580,
            'NFLASH460': 60.289362,
            'SEPIA180': 53.666667,
            'SEPIA345': 54.655556,
            'SEPIA660': 51.776860,
            'HOLO': 41.952381,
            'ZEUS2': 51.393939

        }


        middle = {
            'CHAMP690': 58.285714,
            'HOLO': 41.952381,
            'LASMA345': 55.101010,
            'NFLASH230': 52.413223,
            'NFLASH460': 56.032587,
            'SEPIA180': 52.327957,
            'SEPIA345': 53.891139,
            'SEPIA660': 54.514806,
            'ZEUS2': 51.393939
        }

        if estimateType == 'lower':
            return pd.Timedelta(seconds = lower[rx])
        
        elif estimateType == 'upper':
            return pd.Timedelta(seconds = upper[rx])
        
        elif estimateType == 'middle':
            return pd.Timedelta(seconds = middle[rx])

        else:
            print('Wrong estimateType. Choose between lower, upper and middle')
            return None


def merge_features(paths, all_vals = True):
    """
    Arguments:
        paths: dictionary of paths to csv files
    
    Merges the features from the csv files into one dataframe.
    Can only handle one csv file with ACTUALAZ and ACTUALEL at the moment
    If you introduce more than one with those columns, add suffix to the end
    """

    print(paths, type(paths))
    dfs = {}
    for key in paths:
        dfs[key] = pd.read_csv(paths[key])

    #Merge dfs on date, dont keep duplicate columns
    for i,key in enumerate(dfs):
        dfs[key].dropna         (inplace = True)
        dfs[key].drop_duplicates(inplace = True)
        dfs[key]['date'] = pd.to_datetime(dfs[key]['date'])


    df = pd.DataFrame()
    for key in dfs:
        #merge df and _df
        _df = dfs[key]
        _df = _df.loc[:, ~_df.columns.isin(['ROTATIONZ'])]

        print(key)
        if key.startswith('sumdabs'):
            _df = _df.rename(columns = lambda col: f"{col}_{key}" 
                                            if col not in ('date') 
                                            else col
                        )


        _df = _df.loc[:, ~_df.columns.isin(['WINDSPEED'])]
        #print(df.head())
        
        if df.empty:
            df = _df
        else:
            print(key, _df.columns)
            df = df.merge(_df, left_on='date', right_on='date', how = 'inner')
        #print(df.head())

    print(df.columns)

    if all_vals:
        
        df.to_csv('./Data/merged_features3_all.csv', index = False)
    else:
        df.to_csv('./Data/merged_features3.csv', index = False)

    print('df exported to csv')
    return df


def add_feature():
    """
    Adds feature(s) to df_merged
    
    Not used
    """

    df = pd.read_csv('./Data/merged_features_all.csv')
    df['date'] = pd.to_datetime(df['date'])
    print()
    #print('Number of unique scans:', len(df['scan'].unique()))
    print('Number of scans:', len(df))

    df_pointing = pd.read_csv('./Data/PointingTable.csv')
    df_pointing['obs_date'] = pd.to_datetime(df_pointing['obs_date'])
    df_merged = df.merge(df_pointing.loc[: , ['obs_date', 'rx']], left_on='date', right_on='obs_date', how='left')
    df_merged.dropna(inplace=True)
    df_merged.drop_duplicates(inplace=True)
    #df_merged = pd.concat([df_merged,pd.get_dummies(df_merged['rx'], prefix='rx',dummy_na=True)],axis=1).drop(['rx'],axis=1)

    print(df_merged.head())

    print(df_merged.columns)
    print(df_pointing['rx'].value_counts().sum())
    print(len(df_merged))


def add_features():
    """
    Adds features to df_merged
    

    """

    df = pd.read_csv('./Data/merged_features.csv')
    df['date'] = pd.to_datetime(df['date'])


def all_processing_pipeline():
    """
    This function runs all processing needed to
    obtain a merged_features.csv file.
    Should be used when new data is obtained or if a new workstation is used.
    """

    a = DataProcessing(path_processed_sun)
    a.get_scan_values_sun()

    a = DataProcessing(path_processed_wind)
    a.get_scan_values_wind()

    a = DataProcessing(path_unprocessed_df)
    a.raw_data_processing()
    a.get_scan_values()

    merge_features(path_median, path_median_wind, path_median_sun, all_vals=True)



def raw_data_pipeline(path, dt0, dt1, operation, selected_columns=None, name = 'scans'):
    """
    Pipeline for raw data processing
    """

    a = DataProcessing(path,selected_columns)
    a.raw_data_processing()
    a.get_scan_values(dt0, dt1, operation, name = name)
    return 

path_median         = './Data/' + 'scans_median_0_0_all.csv'
path_median_sun     = './Data/' + 'scans_sun_median_0_0_all.csv'
path_median_wind    = './Data/' + 'scans_wind_median_2_0_all.csv'
path_merged         = './Data/' + 'merged_features_all.csv'
path_sumdabs1       = './Data/' + 'scans_sum_dabs_1_0_all.csv'
path_sumdabs2       = './Data/' + 'scans_sum_dabs_2_0_all.csv'
path_sumdabs5       = './Data/' + 'scans_sum_dabs_5_0_all.csv'
path_dazdel         = './Data/' + 'daz_del_median_0_0_all.csv'

paths = {
    'median'        : path_median,
    'median_sun'    : path_median_sun,
    'median_wind'   : path_median_wind,
    'sumdabs1'      : path_sumdabs1,
    'sumdabs2'      : path_sumdabs2,
    'sumdabs5'      : path_sumdabs5,
    'dazdel'        : path_dazdel
}


turbulence = ['WINDSPEED']
sun = ['DSUNAZ', 'ANGLETOSUN']
pointingAndTilts = ['ACTUALAZ', 'ACTUALEL', 'TILT1X', 'TILT1Y', 'TILT2X', 'TILT2Y', 'TILT3X', 'TILT3Y', 'POSITIONX', 'POSITIONY', 'POSITIONZ', 'DISP_ABS1', 'DISP_ABS2', 'DISP_ABS3']
temperatures = ['TEMPERATURE', 'TEMP1', 'TEMP2', 'TEMP3','TEMP4', 'TEMP5', 'TEMP6', 'TEMP26', 'TEMP27', 'TEMP28', 'TILT1T', 'TILT2T', 'TILT3T']
correction_cols = ['DAZ_DISP','DAZ_SPEM','DAZ_TEMP','DAZ_TILT','DAZ_TILTTEMP','DAZ_TOTAL','DEL_DISP','DEL_SPEM','DEL_TEMP','DEL_TILT','DEL_TILTTEMP','DEL_TOTAL']
weather = ['DEWPOINT','HUMIDITY','PRESSURE']
new_features = ['ACTUALVELOCITYAZ', 'ACTUALVELOCITYEL']
scanValuesParameters = {
    1:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'MEDIAN'   , 'useAllScans': True, 'name': 'ALL', 'estimateType': 'lower', 'selectedColumns': None},
    2:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'MEDIAN'   , 'useAllScans': True, 'name': 'ALL', 'estimateType': 'upper', 'selectedColumns': None},

    3:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    4:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'upper', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    5:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    6:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'upper', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    7:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    8:  {'timeBeforeScan': 0,  'duringScan': True , 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'upper', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
  
    9:  {'timeBeforeScan': 1,  'duringScan': False, 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    10: {'timeBeforeScan': 1,  'duringScan': False, 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    11: {'timeBeforeScan': 1,  'duringScan': False, 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
 
    12: {'timeBeforeScan': 2,  'duringScan': False, 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    13: {'timeBeforeScan': 2,  'duringScan': False, 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    14: {'timeBeforeScan': 2,  'duringScan': False, 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
 
    15: {'timeBeforeScan': 3,  'duringScan': False, 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    16: {'timeBeforeScan': 3,  'duringScan': False, 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    17: {'timeBeforeScan': 3,  'duringScan': False, 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},

    21: {'timeBeforeScan': 4,  'duringScan': False, 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    22: {'timeBeforeScan': 4,  'duringScan': False, 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    23: {'timeBeforeScan': 4,  'duringScan': False, 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},

    24: {'timeBeforeScan': 5,  'duringScan': False, 'operation': 'VAR'      , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    25: {'timeBeforeScan': 5,  'duringScan': False, 'operation': 'SUM_DABS' , 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},
    26: {'timeBeforeScan': 5,  'duringScan': False, 'operation': 'MEAN_DABS', 'useAllScans': True, 'name': 'Tur_PaT_CC', 'estimateType': 'lower', 'selectedColumns': turbulence+pointingAndTilts+correction_cols},

    18: {'timeBeforeScan': 15, 'duringScan': False, 'operation': 'CHANGE'   , 'useAllScans': True, 'name': 'Tmp_Tur_S_W', 'estimateType': 'lower', 'selectedColumns': temperatures+sun+weather+turbulence},
    19: {'timeBeforeScan': 10, 'duringScan': False, 'operation': 'CHANGE'   , 'useAllScans': True, 'name': 'Tmp_Tur_S_W', 'estimateType': 'lower', 'selectedColumns': temperatures+sun+weather+turbulence},
    20: {'timeBeforeScan': 5 , 'duringScan': False, 'operation': 'CHANGE'   , 'useAllScans': True, 'name': 'Tmp_Tur_S_W', 'estimateType': 'lower', 'selectedColumns': temperatures+sun+weather+turbulence},

    27: {'timeBeforeScan': 5, 'duringScan': True, 'operation': 'VAR'   , 'name': 'Tmp_Tur_S_W', 'estimateType': 'lower', 'selectedColumns': turbulence},
    28: {'timeBeforeScan': 5, 'duringScan': True, 'operation': 'CHANGE'   , 'name': 'Tmp_Tur_S_W', 'estimateType': 'lower', 'selectedColumns': ['WINDDIRECTION']},
}




def get_sun_data():
    path = './Data/db_exports/'

    az = pd.read_csv(path + 'ACTUALAZ.csv')
    el = pd.read_csv(path + 'ACTUALEL.csv')

    az['date'] = pd.to_datetime(az['date'])
    el['date'] = pd.to_datetime(el['date'])
    
    df = pd.merge_asof(az, el, on='date', direction='nearest')
    df.loc[df.ACTUALAZ > np.pi, 'ACTUALAZ'] -= 2*np.pi 
    df.loc[df.ACTUALAZ < -np.pi, 'ACTUALAZ'] += 2*np.pi

    #Get sun positions from get_sun_positions function
    az_sun, el_sun = get_sun_position(df['date'])
    #Sun is on interval 0-2pi and az is -pi-pi, turn sun to the same
    az_sun = az_sun % (2*np.pi)
    az_sun[az_sun >= np.pi] -= 2*np.pi

    df_sunaz = pd.DataFrame({'date': df['date'], 'SUNAZ': az_sun})
    df_sunel = pd.DataFrame({'date': df['date'], 'SUNEL': el_sun})

    df_dsunaz = pd.DataFrame({'date': df['date'], 'DSUNAZ': df['ACTUALAZ'] - az_sun})
    df_dsunaz.loc[df_dsunaz["DSUNAZ"] >= np.pi, "DSUNAZ"] -= 2*np.pi
    df_dsunaz.loc[df_dsunaz["DSUNAZ"] <= -np.pi, "DSUNAZ"] += 2*np.pi

    angletosun = calcSmallestAngle(df['ACTUALAZ'], df['ACTUALEL'], az_sun, el_sun)
    df_angletosun = pd.DataFrame({'date': df['date'], 'ANGLETOSUN': angletosun})

    #We dont care about elevation under horizon
    df_sunel.loc[df_sunel["SUNEL"] < 0, "SUNEL"] = 0

    #Write all dataframes to file
    df_sunaz.to_csv(path + 'SUNAZ.csv', index=False)
    df_sunel.to_csv(path + 'SUNEL.csv', index=False)
    df_dsunaz.to_csv(path + 'DSUNAZ.csv', index=False)
    df_angletosun.to_csv(path + 'ANGLETOSUN.csv', index=False)

def get_wind_data():
    path = './Data/db_exports/'
    wind = pd.read_csv(path + 'WINDDIRECTION.csv')
    wind['date'] = pd.to_datetime(wind['date'])
    wind = wind[wind.WINDDIRECTION != 999]
    wind['WINDDIRECTION'] = np.deg2rad(wind['WINDDIRECTION'])
    wind.loc[wind["WINDDIRECTION"] >= np.pi, "WINDDIRECTION"] -= 2*np.pi


    az = pd.read_csv(path + 'ACTUALAZ.csv')
    az['date'] = pd.to_datetime(az['date'])

    df_merged = pd.merge_asof(wind, az, on='date', direction='nearest')
    df_merged[df_merged.ACTUALAZ > np.pi] -= 2*np.pi
    df_merged[df_merged.ACTUALAZ < -np.pi] += 2*np.pi

    df_merged['DWINDDIRECTION'] = df_merged['WINDDIRECTION'] - df_merged['ACTUALAZ']
    df_merged.loc[df_merged["DWINDDIRECTION"] >= np.pi, "DWINDDIRECTION"] -= 2*np.pi
    df_merged.loc[df_merged["DWINDDIRECTION"] <= -np.pi, "DWINDDIRECTION"] += 2*np.pi

    df_merged[['date', 'DWINDDRECTION']].to_csv(path + 'DWINDDIRECTION.csv', index=False)

medians = ['ACTUALAZ', 'ACTUALEL', 'WINDDIRECTION', 'WINDSPEED', 'DAZ_DISP',
            'DAZ_SPEM', 'DAZ_TEMP', 'DAZ_TILT', 'DAZ_TILTTEMP', 'COMMANDAZ', 'COMMANDEL',
            'DEL_SPEM', 'DEL_DISP', 'DEL_TEMP', 'DEL_TILT', 'DEL_TILTTEMP', 'SUNAZ', 'SUNEL', 'TILT1T', 'TILT1X', 'TILT1Y', 'TILT2T', 'TILT2X', 'TILT2Y', 'TILT3T', 'TILT3X', 'TILT3Y',
            'TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'TEMP5', 'TEMP6', 'TEMP26', 'TEMP27', 'TEMP28', 'TEMPERATURE', 'POSITIONX', 'POSITIONY', 'POSITIONZ', 'ROTATIONX', 'ROTATIONY',
            'HUMIDITY', 'DEWPOINT', 'PRESSURE', 'DEL_TOTAL', 'DAZ_TOTAL', 'DISP_ABS1', 'DISP_ABS2', 'DISP_ABS3'
            ]

medians_optical = ['WINDDIRECTION', 'DAZ_DISP',
            'DAZ_SPEM', 'DAZ_TEMP', 'DAZ_TILT', 'DAZ_TILTTEMP', 'DEL_SPEM', 'DEL_DISP', 'DEL_TEMP', 'DEL_TILT', 'DEL_TILTTEMP', 'SUNAZ', 'SUNEL', 'TILT1T', 'TILT1X', 'TILT1Y', 'TILT2T', 'TILT2X', 'TILT2Y', 'TILT3T', 'TILT3X', 'TILT3Y',
            'TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'TEMP5', 'TEMP6', 'TEMP26', 'TEMP27', 'TEMP28', 'TEMPERATURE', 'POSITIONX', 'POSITIONY', 'POSITIONZ', 'ROTATIONX', 'ROTATIONY',
            'HUMIDITY', 'DEWPOINT', 'PRESSURE', 'DEL_TOTAL', 'DAZ_TOTAL', 'DISP_ABS1', 'DISP_ABS2', 'DISP_ABS3'
            ]
other_optical = [
    'WINDSPEED'
]

change_in_medians = ['WINDDIRECTION', 'TEMP1', 'TEMP26', 'TILT1T', 'COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL', 'DAZ_DISP', 'DAZ_SPEM', 'DAZ_TEMP', 'DAZ_TILT', 'DAZ_TILTTEMP', 'DEL_SPEM', 'DEL_DISP', 'DEL_TEMP', 'DEL_TILT', 'DEL_TILTTEMP']
#Feature for regualr offset prediction
feature_params = {
    1: {'operation': 'CHANGE_IN_MEDIAN', 'suffix': 'CHANGE_I1', 'columns': change_in_medians, 'interval': 1},
    2: {'operation': 'INTEGRATE_POSITIVE_CHANGE', 'suffix': 'POS_CHANGE', 'columns': ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']},
    3: {'operation': 'INTEGRATE_NEGATIVE_CHANGE', 'suffix': 'NEG_CHANGE', 'columns': ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']},
    4: {'operation': 'MEDIAN', 'suffix': 'MEDIAN', 'columns': medians, 'during': True},
    5: {'operation': 'VARIANCE', 'suffix': 'VARIANCE_P5', 'columns': ['WINDSPEED'], 'period': 5},
    6: {'operation': 'MAX_CHANGE_IN_INTERVAL', 'suffix': 'MAX_CHANGE_I5', 'columns': ['TEMP1', 'TEMP26'], 'interval': 5},
    7: {'operation': 'MAX', 'suffix': 'MAX', 'columns': ['ACTUALVELOCITYAZ', 'ACTUALVELOCITYEL']},
    8: {'operation': 'CHANGE', 'suffix': 'CHANGE_P5', 'columns': ['DSUNAZ'], 'period': 5}
}

feature_params_optical ={
    # 'variance': {'operation': 'VARIANCE', 'suffix': 'VAR_5', 'columns': other_optical, 'period': 5},
    # 'medians': {'operation': 'MEDIAN', 'suffix': 'MEDIAN_1', 'columns': medians_optical, 'around': 0.5},
    'neg_change': {'operation': 'INTEGRATE_NEGATIVE_CHANGE', 'suffix': 'NEG_CHANGE', 'columns': ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']},
    'pos_change': {'operation': 'INTEGRATE_POSITIVE_CHANGE', 'suffix': 'POS_CHANGE', 'columns': ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']},
    'medians': {'operation': 'MEDIAN', 'suffix': 'MEDIAN_1', 'columns': ['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL', 'WINDSPEED'], 'around': 0.5},
    'change_in_medians': {'operation': 'CHANGE_IN_MEDIAN', 'suffix': 'CHANGE_I1', 'columns': change_in_medians, 'interval': 1},
}

data_frequency = {
    'TEMP1': 0.1666667,
    'TEMP2': 0.1666667,
    'TEMP3': 0.1666667,
    'TEMP4': 0.1666667,
    'TEMP5': 0.1666667,
    'TEMP6': 0.1666667,
    'TEMP26': 0.5,
    'TEMP27': 0.5,
    'TEMP28': 0.5,
    'WINDSPEED': 0.1833333333,
    'WINDDIRECTION': 0.1833333333,
}




def get_features_for_all_datasets(i_process):

    path_ds = './Datasets/'
    datasets = os.listdir(path_ds)
    datasets = [x for x in datasets if x.endswith('.csv')]

    map_dataset_to_params = {
        'tmp2022_clean.csv':[4,5,8] ,
        'tmp2022_clean_nfslash230.csv':[4,5,8] ,
        'tmp2022_clean_clf.csv':[4,5,8] ,
        'tmp2022_clean_clf_nfslash230.csv':[4,5,8] ,
        'tmp2022_clean_transformed.csv':[1,2,3,4,5,6,7,8] ,
        'tmp2022_clean_nflash230_transformed.csv':[1,2,3,4,5,6,7,8] ,
        'tmp2022_clean_clf_transformed.csv':[1,2,3,4,5,6,7,8] ,
        'tmp2022_clean_clf_nflash230_transformed.csv':[1,2,3,4,5,6,7,8] ,
    }

    fn = datasets[i_process].split('.')[0]

    path_output = os.path.join(path_ds,fn)
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    path_output = os.path.join(path_output, 'features.csv')
    params_for_dataset = {key: feature_params[key] for key in map_dataset_to_params[datasets[i_process]]}
    print('Process', i_process)
    print('Path dataset', path_ds + datasets[i_process])
    print('Path output', path_output)
    print('Params for dataset', params_for_dataset)
    processing = DataProcessing(path_scans = path_ds + datasets[i_process])
    processing.get_scan_values_v3(feature_params = params_for_dataset, PATH_OUTPUT = path_output)

    print('All done')



if __name__=='__main__':
    # selected_columns = ['date', 'ACTUALAZ', 'ACTUALEL','TILT1X','TILT1Y','POSITIONX','POSITIONY','POSITIONZ','ROTATIONX','ROTATIONY']
    
    #raw_data_pipeline(path_unprocessed_df, 0,0, 'median', correction_cols, name = 'daz_del')
    # raw_data_pipeline(path_unprocessed_df, 1,0,'sum_dabs', selected_columns)
    # raw_data_pipeline(path_unprocessed_df, 2,0,'sum_dabs', selected_columns)
    # raw_data_pipeline(path_unprocessed_df, 5,0,'sum_dabs', selected_columns)
    # test = DataProcessing(path_df = './Data/raw/raw_data_15_10.csv')
    # test.raw_data_processing()
    #Read df from Data/processed_v4/all_features.csv , fk_4_1.csv and fk_4_2.csv and join them together
    # df1 = pd.read_csv('./Data/processed_v4/all_features.csv')
    # df2 = pd.read_csv('./Data/processed_v4/fk_4_1.csv')
    # df3 = pd.read_csv('./Data/processed_v4/fk_4_2.csv')
    # df_merged = pd.concat([df1, df2.loc[: , df2.columns != 'date'], df3.loc[: , df3.columns != 'date']], axis = 1)

    # get_features_for_all_datasets(i_process = int(sys.argv[1]))

    test2 = DataProcessing(path_scans='./Data/raw_nflash230.csv')
    test2.get_scan_values_optical(feature_params_optical)
    
    """
    df = pd.read_csv('./Data/processed_v3/all_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df2 = pd.read_csv('./Data/processed_v3/all_features_safe.csv')
    df2['date'] = pd.to_datetime(df2['date'])
    embed()
    """
    #raw_data_pipeline(path_unprocessed_df, -0.5,-0.5, 'median', name = 'scans_shorter_int')
    #merge_features(paths)
    pass


