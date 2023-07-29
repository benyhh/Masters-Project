import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer,OneHotEncoder, LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import mean_squared_error, r2_score
import importlib
import random
import settings
importlib.reload(settings)
from settings import patches, features, dataset_params
from IPython import embed
import os
random_seed = 412069413
from clean_pointing_table import clean_pointing_table
import pickle


class PrepareDataNN():
    def __init__(self,
                df,
                features = None,
                run_number = None,
                ) -> None:


        # embed(header = 'combined ds')
        X = df[features]
        y = df[['OFFSETAZ', 'OFFSETEL']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.scale_data()

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # embed(header = 'scale data')
        self.X_train = self.X_scaler.fit_transform(self.X_train)
        self.X_test = self.X_scaler.transform(self.X_test)
        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.y_test = self.y_scaler.transform(self.y_test)

    def rescale_y(self, y):
        return self.y_scaler.inverse_transform(y)


class PrepareDataCombined():
    def __init__(self,
                df,
                nonlinear_features = None,
                linear_features = None,
                run_number = None,
                scale_data = True
                ) -> None:


        # embed(header = 'combined ds')
        X_linear = df[linear_features]
        X_nonlinear = df[nonlinear_features]

        y = df[['OFFSETAZ', 'OFFSETEL']]

        X_linear_train, X_linear_test, y_train, y_test = train_test_split(X_linear, y, test_size=0.25, random_state=42)

        X_nonlinear_train = df.loc[X_linear_train.index][nonlinear_features]
        X_nonlinear_test = df.loc[X_linear_test.index][nonlinear_features]

        self.X_linear_train = X_linear_train
        self.X_linear_test = X_linear_test
        self.X_nonlinear_train = X_nonlinear_train
        self.X_nonlinear_test = X_nonlinear_test
        self.y_train = y_train
        self.y_test = y_test

        if scale_data:
            self.scale_data()
        else:
            self.X_linear_train = X_linear_train.values
            self.X_linear_test = X_linear_test.values
            self.X_nonlinear_train = X_nonlinear_train.values
            self.X_nonlinear_test = X_nonlinear_test.values
            self.y_train = y_train.values
            self.y_test = y_test.values

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        
        self.lin_scaler = StandardScaler()
        self.nonlin_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.X_linear_train = self.lin_scaler.fit_transform(self.X_linear_train)
        self.X_linear_test = self.lin_scaler.transform(self.X_linear_test)
        self.X_nonlinear_train = self.nonlin_scaler.fit_transform(self.X_nonlinear_train)
        self.X_nonlinear_test = self.nonlin_scaler.transform(self.X_nonlinear_test)
        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.y_test = self.y_scaler.transform(self.y_test)

    def rescale_y(self, y):
        return self.y_scaler.inverse_transform(y)


class PrepareDataFinal():
    def __init__(self,
                df,
                parameter_keys = None,
                feature_list = None,
                target_key = None,
                run_number = None,
                split_on_days = False,
                ) -> None:

        targets = {
            'az'   : ['Off_Az'],
            'el'   : ['Off_El'],
        }

        self.params = None
        self.target = targets[target_key]
        self.n_targets = len(self.target)
        self.dataset_key, self.timeperiod_key = parameter_keys
        self.run_number = run_number 

        # if target_key == 'az':
        #     df = df.loc[: , df.columns != 'Off_El']
        # elif target_key == 'el':
        #     df = df.loc[: , df.columns != 'Off_Az']
        
        if 'rx' in df.columns:
            le = LabelEncoder()
            df['rx'] = le.fit_transform(df['rx'])
        
        self.df = df

        if split_on_days:
            self.train_test_split_days()
        else:
            self.df_train, self.df_test = train_test_split(df, test_size=0.4, random_state=random_seed)

        self.X_train = self.df_train[feature_list]
        self.X_test = self.df_test[feature_list]
        self.y_train = self.df_train[self.target]
        self.y_test = self.df_test[self.target]

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.25, random_state=random_seed)

        self.rms_offset = np.sqrt(np.mean(self.y_test**2)).get(0)
        self.rms_offset_optimal_correction = np.sqrt( np.mean( (df[self.target] - df[self.target].mean())**2 ) ).get(0)


    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
 
    def train_test_split_days(self):

        df = self.df
        df.insert(0, 'day', df['date'].dt.date)
        dfs = [df[df['day'] == day] for day in df['day'].unique()]
        random.Random(random_seed).shuffle(dfs)

        test_size = 0.33
        train_size = 1 - test_size
        n_days     = len(dfs)

        dfs_train = dfs[:int(train_size * n_days)]
        dfs_test  = dfs[int(train_size * n_days):]

        self.df_train = pd.concat(dfs_train)
        self.df_test  = pd.concat(dfs_test)

        self.df_train = self.df_train.loc[: , self.df_train.columns != 'day']
        self.df_test  = self.df_test.loc [: , self.df_test.columns  != 'day']
        train_days = len(self.df_train)
        test_days  = len(self.df_test)
        print(f'Training days: {train_days} | Test days: {test_days} | Test size: {test_days/(train_days+test_days):.2f}')
        return


class PrepareDataForNewModel():
    def __init__(self,
                df_path              = './Data/processed_v4/all_features_safe.csv',
                params               = dataset_params
                ) -> None:

        targets = {
            "total": ["Offset"],
            "az"   : ["Off_Az_new"],
            "el"   : ["Off_El_new"],
            "both" : ["Off_El_new", "Off_Az_new"],
            'eel'  : ['Off_eEl'],
            'eaz'  : ['Off_eAz'],
            'real_az' : ['REALAZ'],
            'real_el' : ['REALEL'],
            'residual_az' : ['RESIDUALAZ'],
            'residual_el' : ['RESIDUALEL'],
            'optical_az' : ['ACTUALAZ'],
            'optical_el' : ['ACTUALEL'],
            'optical_both' : ['ACTUALAZ', 'ACTUALEL'],
            }



        self.params = params
        self.scaled = False
        self.target = targets[params['target']]
        self.n_targets = len(self.target)
        self.df = pd.read_csv(df_path)

        if params['optical_model']:
            self.df[features[params['feature_key']] + self.target] = np.deg2rad(self.df[features[params['feature_key']] + self.target])

        self.df = self.df.drop_duplicates(keep = 'first')
        self.df = self.df.dropna()

        df_train,df_test = train_test_split(self.df, test_size=0.2, random_state=random_seed)

        self.X_train = df_train[features[params['feature_key']]]
        self.X_test = df_test[features[params['feature_key']]]
        self.y_train = df_train[self.target]
        self.y_test = df_test[self.target]
        

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.25, random_state=random_seed)

        if params['new_model'] == 'residual_az' or params['target'] == 'real_az':
            self.benchmark = df_test.Off_Az.abs().mean()
        elif params['target'] == 'residual_el' or params['target'] == 'real_el':
            self.benchmark = df_test.Off_El.abs().mean()
        else:
            self.benchmark = 0

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test




class PrepareDataRaw_v2():
    """
    Prepares data for model to learn the underlying analytical pointing model.
    Inputs are COMMANDAZ and COMMANDEL, and outputs are ACTUALAZ or ACTUALEL.
    """
    def __init__(self,
                target_key = 'both', #az, el or both
                use_cartesian = False,
                use_scaler = False,
                run_number = None,
                ) -> None:

        targets_dict = {
            'az': ['ACTUALAZ'],
            'el': ['ACTUALEL'],
            'both': ['ACTUALAZ', 'ACTUALEL']
        }
        target = targets_dict[target_key]
        self.target = target
        self.n_targets = len(target)

        self.params = None # No params for this class


        df = pd.read_csv('./Data/raw_nflash230.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] < pd.Timestamp('2022-09-17')]
        df = np.deg2rad(df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])
        terms_az, terms_el, terms_both = self.prepare_linear_terms(df['COMMANDAZ'], df['COMMANDEL'])
        self.df = df

        self.benchmark = 0

        if target_key == 'both':
            X = np.column_stack(terms_both)
            y = df[['ACTUALAZ', 'ACTUALEL']]
    
        elif target_key == 'az':
            X = np.column_stack(terms_both)
            y = df[['ACTUALAZ']]

        elif target_key == 'el':
            X = np.column_stack(terms_both)
            y = df[['ACTUALEL']]



        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        if use_scaler:
            self.X_train, self.X_test, self.y_train, self.y_test = self.scale_data(self.X_train, self.X_test, self.y_train, self.y_test)

        self.feature_names = [str(i) for i in range(self.X_train.shape[1])]

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def prepare_linear_terms(self, Az,El):
        terms_az = [
            Az,
            El,
            np.sin(Az),
            np.cos(3*Az),
            np.sin(2*Az),
            np.cos(2*Az),
            np.cos(Az) * np.tan(El),
            np.sin(Az) * np.tan(El),
            np.tan(El),
            1/np.cos(El),
            np.cos(2*Az) / np.cos(El),
            np.cos(Az) / np.cos(El),
            np.cos(5*Az) / np.cos(El),
            # np.ones(len(Az)),
        ]
        terms_el = [
            Az,
            El,
            np.sin(El),
            np.cos(El),
            np.cos(2*Az),
            np.sin(2*Az),
            np.cos(3*Az),
            np.sin(3*Az),
            np.sin(4*Az),
            np.sin(5*Az),
            np.sin(Az), #AW
            np.sin(Az)*np.tan(El),
            np.cos(Az), #AN
            # np.ones(len(Az)),
        ]

        #Set of all terms in terma_az and terms_el
        terms_both = [
            Az,
            El,
            # np.sin(Az), #AW EL
            # np.cos(3*Az),
            # np.sin(2*Az),
            # np.cos(2*Az),
            # np.cos(Az) * np.tan(El), #AW AZ
            # np.sin(Az) * np.tan(El), #AN AZ,
            # np.cos(Az), #AN EL
            # np.tan(El), #NPAE
            # 1/np.cos(El), #CA
            # np.ones(len(Az)), 
            # np.sin(El),
            # np.cos(El),
            # np.sin(3*Az),
            # np.sin(4*Az),
            # np.sin(5*Az),
            # np.cos(2*Az) / np.cos(El),
            # np.cos(Az) / np.cos(El),
            # np.cos(5*Az) / np.cos(El)
        ]

        
        return terms_az, terms_el, terms_both  


    def use_cartesian(self):
        """
        Transforms data with az and el to cartesian coordinates
        """

        df = self.df

        if 'ACTUALAZ' in df.columns and 'ACTUALEL' in df.columns:
            x = np.sin(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            y = np.cos(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            z = np.sin(df['ACTUALEL'].values)

            df.insert(0, 'ACTUALX', x)
            df.insert(0, 'ACTUALY', y)
            df.insert(0, 'ACTUALZ', z)        
        
        if 'COMMANDAZ' in df.columns and 'COMMANDEL' in df.columns:
            x = np.sin(df['COMMANDAZ'].values) * np.cos(df['COMMANDEL'].values)
            y = np.cos(df['COMMANDAZ'].values) * np.cos(df['COMMANDEL'].values)
            z = np.sin(df['COMMANDEL'].values)

            df.insert(0, 'COMMANDX', x)
            df.insert(0, 'COMMANDY', y)
            df.insert(0, 'COMMANDZ', z)    
        
        old_cols = ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']
        df = df.loc[ : , ~df.columns.isin(old_cols)]
        
        self.df = df


    def scale_data(self, X_train, X_test, y_train, y_test):
        #Scale data with standard scaler and make it works even though y have multiple columns
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train = self.X_scaler.fit_transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        if self.n_targets == 2:
            y_train = self.y_scaler.fit_transform(y_train)
            y_test = self.y_scaler.transform(y_test)
        else:
            y_train = self.y_scaler.fit_transform(y_train).reshape(-1)
            y_test = self.y_scaler.transform(y_test).reshape(-1)

        return X_train, X_test, y_train, y_test

    def rescale_data(self, y):
        if self.n_targets == 2:
            y = self.y_scaler.inverse_transform(y)
        else:
            y = self.y_scaler.inverse_transform(y).reshape(-1)
        return y



class PrepareDataRaw():
    """
    Prepares data for model to learn the underlying analytical pointing model.
    Inputs are COMMANDAZ and COMMANDEL, and outputs are ACTUALAZ or ACTUALEL.
    """
    def __init__(self,
                
                path_features = None,
                target_key = 'both', #az, el or both
                use_cartesian = False,
                use_scaler = True,
                params = None
                ) -> None:

        targets_dict = {
            'az': ['ACTUALAZ'],
            'el': ['ACTUALEL'],
            'both': ['ACTUALAZ', 'ACTUALEL']
        }
        target = targets_dict[target_key]
        self.target = target
        self.n_targets = len(target)

        self.params = params # No params for this class

        self.df = pd.read_csv('./Data/raw_nflash230.csv')

        self.df.drop_duplicates(inplace = True)

        if path_features is not None:
            self.df_features = pd.read_csv(path_features)
            #Use selected columns specified in settings.py
            self.df_features = self.df_features[['date'] + features[ params['feature_key'] ]]
            #Left join df and df_features on date column
            self.df = pd.merge(self.df, self.df_features, on='date', how='left')
            self.df = self.df.dropna()
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df[self.df.date < pd.Timestamp('2022-09-17')]
        self.df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']] = np.deg2rad(self.df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])

        if use_cartesian:
            self.use_cartesian()
            X = self.df[['COMMANDX', 'COMMANDY', 'COMMANDZ']]
            y = self.df[['ACTUALX', 'ACTUALY', 'ACTUALZ']]
        else:
            X = self.df.loc[: , ~self.df.columns.isin(['date', 'source','scan','source'] + target)]
            y = self.df[target]

        print(self.df.describe())
        #Print column name and index for all columns in X and y
        print('Feature columns')
        for i, col in enumerate(X.columns):
            print(i, col)
        print('Target columns')
        for i, col in enumerate(y.columns):
            print(i, col)
    

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed)

        if use_scaler:
            self.X_train, self.X_test, self.y_train, self.y_test = self.scale_data(self.X_train, self.X_test, self.y_train, self.y_test)

        self.feature_names = X.columns

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


    def use_cartesian(self):
        """
        Transforms data with az and el to cartesian coordinates
        """

        df = self.df

        if 'ACTUALAZ' in df.columns and 'ACTUALEL' in df.columns:
            x = np.sin(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            y = np.cos(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            z = np.sin(df['ACTUALEL'].values)

            df.insert(0, 'ACTUALX', x)
            df.insert(0, 'ACTUALY', y)
            df.insert(0, 'ACTUALZ', z)        
        
        if 'COMMANDAZ' in df.columns and 'COMMANDEL' in df.columns:
            x = np.sin(df['COMMANDAZ'].values) * np.cos(df['COMMANDEL'].values)
            y = np.cos(df['COMMANDAZ'].values) * np.cos(df['COMMANDEL'].values)
            z = np.sin(df['COMMANDEL'].values)

            df.insert(0, 'COMMANDX', x)
            df.insert(0, 'COMMANDY', y)
            df.insert(0, 'COMMANDZ', z)    
        
        old_cols = ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']
        df = df.loc[ : , ~df.columns.isin(old_cols)]
        
        self.df = df



    def scale_data(self, X_train, X_test, y_train, y_test):
        #Scale data with standard scaler and make it works even though y have multiple columns
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train = self.X_scaler.fit_transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        if self.n_targets == 2:
            y_train = self.y_scaler.fit_transform(y_train)
            y_test = self.y_scaler.transform(y_test)
        else:
            y_train = self.y_scaler.fit_transform(y_train).reshape(-1)
            y_test = self.y_scaler.transform(y_test).reshape(-1)

        self.save_scaler()
        return X_train, X_test, y_train, y_test

    def rescale_data(self, y):
        if self.n_targets == 2:
            y = self.y_scaler.inverse_transform(y)
        else:
            y = self.y_scaler.inverse_transform(y).reshape(-1)
        return y

    def save_scaler(self):
        PATH_SCALER = f'./AnalyticalModelRaw/Scaler/'
        
        with open(PATH_SCALER + f'X_scaler_{self.params["name"]}.pkl', 'wb') as f:
            pickle.dump(self.X_scaler, f)
        
        with open(PATH_SCALER + f'y_scaler_{self.params["name"]}.pkl', 'wb') as f:
            pickle.dump(self.y_scaler, f)
        return

class PrepareDataAnalytical_v3():
    def __init__(self,
                target_key = 'both', #az, el or both
                use_cartesian = False,
                use_time_filter = True,
                ) -> None:

        targets_dict = {
            'az': ['TOTALAZ'],
            'el': ['TOTALEL'],
            'both': ['TOTALAZ', 'TOTALEL'],
            'testaz': ['ACTUALAZ'],
            'testel': ['ACTUALEL'],
        }
        target = targets_dict[target_key]
        self.target = target
        self.n_targets = len(target)
        self.target_key = target_key
        self.params = None # No params for this class
        startTime = pd.Timestamp('2022-05-23')
        endTime = pd.Timestamp('2022-07-04')

        main_columns = ['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']
        dfs = []
        for column in main_columns:
            df_tmp = pd.read_csv(f'./Data/db_exports/{column}.csv')
            df_tmp['date'] = pd.to_datetime(df_tmp['date'])
            df_tmp = df_tmp[(df_tmp['date'] >= startTime) & (df_tmp['date'] <= endTime)]
            dfs.append(df_tmp)
        
        df = dfs[0]
        for df_tmp in dfs[1:]:
            df = pd.merge(df, df_tmp, how = 'inner', on = ['date'])
               
        features = ['TEMP1']#, 'HUMIDITY','POSITIONZ', 'SUNAZ', 'SUNEL','TILT1X','TILT1Y' ]
        
        for feature in features:
            df_tmp = pd.read_csv(f'./Data/db_exports/{feature}.csv')
            df_tmp['date'] = pd.to_datetime(df_tmp['date'])
            df_tmp = df_tmp[(df_tmp['date'] >= startTime) & (df_tmp['date'] <= endTime)]
            df = pd.merge_asof(df, df_tmp, on='date', direction='backward')

        self.df = df 
        if use_time_filter:
            self.filter_timeintervals()

        self.df['diffaz'] = self.df.COMMANDAZ - self.df.ACTUALAZ
        self.df['diffel'] = self.df.COMMANDEL - self.df.ACTUALEL

        self.df['diffdiffaz'] = np.abs(self.df.COMMANDAZ.diff() - self.df.ACTUALAZ.diff())
        self.df['diffdiffel'] = np.abs(self.df.COMMANDEL.diff() - self.df.ACTUALEL.diff())

        #smallest angle between vector in radians COMMANDAZ COMMANDEL and ACTUALAZ ACTUALEL
        df = self.df.copy()
        dotprod = np.cos(df.COMMANDAZ) * np.cos(df.COMMANDEL) * np.cos(df.ACTUALAZ) * np.cos(df.ACTUALEL) + np.sin(df.COMMANDAZ) * np.cos(df.COMMANDEL) * np.sin(df.ACTUALAZ) * np.cos(df.ACTUALEL) + np.sin(df.COMMANDEL) * np.sin(df.ACTUALEL)
        df['angle'] = np.arccos(dotprod)
        self.df = df.copy()
        
        
        #COMMANDAZ and COMMANDEL are altaz coordinates, calculate the distance between COMMAND and ACTUAL

        # for col in ['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL', 'diffaz', 'diffel', 'angle', 'diffdiffaz', 'diffdiffel']:
        #     self.df[col] = np.rad2deg(self.df[col])


        thresholdaz = self.df['diffdiffaz'].quantile(0.75)
        thresholdel = self.df['diffdiffel'].quantile(0.75)

        self.df = self.df.loc[(self.df['diffdiffaz'] < thresholdaz) & (self.df['diffdiffel'] < thresholdel)]
        
        # embed()
        # self.df = self.df.loc[(np.abs(self.df['diffaz']) < 0.01) & (np.abs(self.df['diffel']) < 0.01)]
        df_pointing = pd.read_csv(f'./Data/tmp2022_cleanedRules.csv')
        #rename obs_date to date
        df_pointing.rename(columns={'obs_date':'date'}, inplace=True)
        df_pointing['date'] = pd.to_datetime(df_pointing['date'])
        # df_pointing = df_pointing.loc[df_pointing.rx == 'NFLASH460']


        self.df = pd.merge_asof(self.df, df_pointing[['date', 'ca','ie','Off_Az','Off_El','freq']], on='date', direction='backward')

        # self.df['ACTUALAZ'] = self.rad2arcsecs(self.df['ACTUALAZ'])
        # self.df['ACTUALEL'] = self.rad2arcsecs(self.df['ACTUALEL'])
        self.df['TOTALAZ'] = self.df['ACTUALAZ'] + self.df['Off_Az']
        self.df['TOTALEL'] = self.df['ACTUALEL'] + self.df['Off_El']

        self.df['ACTUALEL'] = self.df['ACTUALEL'] - np.deg2rad(self.df['ie']/3600)
        self.df['ACTUALAZ'] = self.df['ACTUALAZ'] - np.deg2rad(self.df['ca']/3600) / np.cos(self.df['ACTUALEL'])
        
        # X = self.df[features + ['COMMANDAZ', 'COMMANDEL', 'ca', 'ie']]
        # y = self.df[target]
        X = self.df[['COMMANDAZ', 'COMMANDEL', 'freq']]
        y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)





    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


    def filter_timeintervals(self):

        time_intervals = [
            (pd.Timestamp('2022-05-22 13'), pd.Timestamp('2022-05-22 22')),
            (pd.Timestamp('2022-05-24 11'), pd.Timestamp('2022-05-24 16')),
            (pd.Timestamp('2022-05-25 06'), pd.Timestamp('2022-05-25 17')),
            (pd.Timestamp('2022-06-13 02'), pd.Timestamp('2022-06-13 19')),
        ]
        # time_intervals = [
        #     (pd.Timestamp('2022-05-23 04'), pd.Timestamp('2022-05-23 14')),
        #     (pd.Timestamp('2022-05-24 08'), pd.Timestamp('2022-05-24 11')),
        #     #(pd.Timestamp('2022-05-26 04'), pd.Timestamp('2022-05-27 15')),
        #     (pd.Timestamp('2022-06-21 04'), pd.Timestamp('2022-06-21 08')),
        # ]
        time_intervals = [
            (pd.Timestamp('2022-05-22 09:38:11'), pd.Timestamp('2022-05-23 15:35:45')),
            (pd.Timestamp('2022-05-23 23:03:26'), pd.Timestamp('2022-05-24 14:07:59')),
            (pd.Timestamp('2022-05-25 13:18:19'), pd.Timestamp('2022-05-25 14:30:29')),
            (pd.Timestamp('2022-05-25 17:22:40'), pd.Timestamp('2022-05-25 17:52:27')),
            (pd.Timestamp('2022-06-13 08:27:35'), pd.Timestamp('2022-06-13 14:41:44')),
            (pd.Timestamp('2022-06-13 15:48:00'), pd.Timestamp('2022-06-13 16:48:31')),
            (pd.Timestamp('2022-06-19 11:17:52'), pd.Timestamp('2022-06-19 11:26:44')),
            (pd.Timestamp('2022-06-21 02:50:20'), pd.Timestamp('2022-06-21 04:25:02')),
            (pd.Timestamp('2022-06-21 04:25:02'), pd.Timestamp('2022-06-21 04:25:02')),   
        ]


        interval_masks = [((self.df['date'] >= start) & (self.df['date'] <= end)) for start, end in time_intervals]
        combined_mask = pd.concat(interval_masks, axis=1).any(axis=1)

        # filter the dataframe using the combined mask
        self.df = self.df[combined_mask]

    def rad2arcsecs(self,series):
        #Convert series from radians to arcseconds
        series = series * 3600 * 180 / np.pi
        return series



class PrepareDataAnalytical_v2():
    """
    Prepares data for model to learn the underlying analytical pointing model.
    Inputs are COMMANDAZ and COMMANDEL, and outputs are ACTUALAZ or ACTUALEL.
    Have to subtract ca from ACTUALAZ and ie from ACTUALEL to get the accurate model
    """
    def __init__(self,
                target_key = 'both', #az, el or both
                use_cartesian = False,
                use_time_filter = False,
                ) -> None:

        targets_dict = {
            'az': ['ACTUALAZ'],
            'el': ['ACTUALEL'],
            'both': ['ACTUALAZ', 'ACTUALEL']
        }
        target = targets_dict[target_key]

        self.params = None # No params for this class

        cmd_az = pd.read_csv('./Data/db_exports/COMMANDAZ.csv')
        cmd_el = pd.read_csv('./Data/db_exports/COMMANDEL.csv')
        act_az = pd.read_csv('./Data/db_exports/ACTUALAZ.csv')
        act_el = pd.read_csv('./Data/db_exports/ACTUALEL.csv')

        startTime = pd.Timestamp('2022-05-23')
        endTime = pd.Timestamp('2022-07-04')

        cmd_az['date'] = pd.to_datetime(cmd_az['date'])
        cmd_el['date'] = pd.to_datetime(cmd_el['date'])
        act_az['date'] = pd.to_datetime(act_az['date'])
        act_el['date'] = pd.to_datetime(act_el['date'])

        cmd_az = cmd_az[(cmd_az['date'] >= startTime) & (cmd_az['date'] <= endTime)]
        cmd_el = cmd_el[(cmd_el['date'] >= startTime) & (cmd_el['date'] <= endTime)]
        act_az = act_az[(act_az['date'] >= startTime) & (act_az['date'] <= endTime)]
        act_el = act_el[(act_el['date'] >= startTime) & (act_el['date'] <= endTime)]

        #Merge into 1 dataframe
        print(len(cmd_az), len(cmd_el), len(act_az), len(act_el))
        # df = pd.concat([cmd_az, cmd_el, act_az, act_el], axis = 1)
        # df = pd.concat([cmd_az, cmd_el['COMMANDEL'], act_az['ACTUALAZ'], act_el['ACTUALEL']], axis = 1)

        df = pd.merge(cmd_az, cmd_el, how = 'inner', on = ['date'])
        df = pd.merge(df, act_az, how = 'inner', on = ['date'])
        df = pd.merge(df, act_el, how = 'inner', on = ['date'])
        self.df = df

        if use_time_filter:
            self.filter_timeintervals()

        if use_cartesian:
            self.use_cartesian()
            X = self.df[['COMMANDX', 'COMMANDY', 'COMMANDZ']]
            y = self.df[['ACTUALX', 'ACTUALY', 'ACTUALZ']]
        else:
            X = self.df[['COMMANDAZ', 'COMMANDEL']]
            y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)

        self.target = target
        self.n_targets = len(target)


    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


    def use_cartesian(self):
        """
        Transforms data with az and el to cartesian coordinates
        """

        df = self.df

        if 'ACTUALAZ' in df.columns and 'ACTUALEL' in df.columns:
            x = np.sin(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            y = np.cos(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            z = np.sin(df['ACTUALEL'].values)

            df.insert(0, 'ACTUALX', x)
            df.insert(0, 'ACTUALY', y)
            df.insert(0, 'ACTUALZ', z)        
        
        if 'COMMANDAZ' in df.columns and 'COMMANDEL' in df.columns:
            x = np.sin(df['COMMANDAZ'].values) * np.cos(df['COMMANDEL'].values)
            y = np.cos(df['COMMANDAZ'].values) * np.cos(df['COMMANDEL'].values)
            z = np.sin(df['COMMANDEL'].values)

            df.insert(0, 'COMMANDX', x)
            df.insert(0, 'COMMANDY', y)
            df.insert(0, 'COMMANDZ', z)    
        
        old_cols = ['ACTUALAZ', 'ACTUALEL', 'COMMANDAZ', 'COMMANDEL']
        df = df.loc[ : , ~df.columns.isin(old_cols)]
        
        self.df = df

    def filter_timeintervals(self):

        time_intervals = [
            (pd.Timestamp('2022-05-22 13'), pd.Timestamp('2022-05-22 22')),
            (pd.Timestamp('2022-05-24 11'), pd.Timestamp('2022-05-24 16')),
            (pd.Timestamp('2022-05-25 06'), pd.Timestamp('2022-05-25 17')),
            (pd.Timestamp('2022-06-13 02'), pd.Timestamp('2022-06-13 19')),
        ]
        time_intervals = [
            (pd.Timestamp('2022-05-23 04'), pd.Timestamp('2022-05-23 14')),
            (pd.Timestamp('2022-05-24 08'), pd.Timestamp('2022-05-24 11')),
            #(pd.Timestamp('2022-05-26 04'), pd.Timestamp('2022-05-27 15')),
            (pd.Timestamp('2022-06-21 04'), pd.Timestamp('2022-06-21 08')),
        ]
        time_intervals = [
            (pd.Timestamp('2022-05-22 09:38:11'), pd.Timestamp('2022-05-23 15:35:45')),
            (pd.Timestamp('2022-05-23 23:03:26'), pd.Timestamp('2022-05-24 14:07:59')),
            (pd.Timestamp('2022-05-25 13:18:19'), pd.Timestamp('2022-05-25 14:30:29')),
            (pd.Timestamp('2022-05-25 17:22:40'), pd.Timestamp('2022-05-25 17:52:27')),
            (pd.Timestamp('2022-06-13 08:27:35'), pd.Timestamp('2022-06-13 14:41:44')),
            (pd.Timestamp('2022-06-13 15:48:00'), pd.Timestamp('2022-06-13 16:48:31')),
            (pd.Timestamp('2022-06-19 11:17:52'), pd.Timestamp('2022-06-19 11:26:44')),
            (pd.Timestamp('2022-06-21 02:50:20'), pd.Timestamp('2022-06-21 04:25:02')),
            (pd.Timestamp('2022-06-21 04:25:02'), pd.Timestamp('2022-06-21 04:25:02')),   
        ]
        
        interval_masks = [((self.df['date'] >= start) & (self.df['date'] <= end)) for start, end in time_intervals]
        combined_mask = pd.concat(interval_masks, axis=1).any(axis=1)

        # filter the dataframe using the combined mask
        self.df = self.df[combined_mask]


class PrepareDataAnalytical():
    """
    Prepares data for model to learn the underlying analytical pointing model.
    Inputs are COMMANDAZ and COMMANDEL, and outputs are ACTUALAZ or ACTUALEL.
    Have to subtract ca from ACTUALAZ and ie from ACTUALEL to get the accurate model
    """
    def __init__(self,
                df_path = './Data/processed_v4/all_features_safe.csv',
                target_key = 'ACTUALEL_MEDIAN', #Actualaz or actualel
                ) -> None:

        targets_dict = {
            'az': ['ACTUALAZ'],
            'el': ['ACTUALEL'],
            'both': ['ACTUALAZ', 'ACTUALEL']
        }
        target = targets_dict[target_key]

        self.params = None # No params for this class
        self.df = pd.read_csv(df_path)
        self.df = self.df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN', 'date'] + target]
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(self.df.describe())
        df_pointing = pd.read_csv('./Data/tmp2022.csv')
        df_pointing['obs_date'] = pd.to_datetime(df_pointing['obs_date'])
        startTimeBool = df_pointing['obs_date'] >= pd.Timestamp('2022-07-05')
        endTimeBool = df_pointing['obs_date'] <= pd.Timestamp('2022-07-25')
        df_pointing = df_pointing[startTimeBool & endTimeBool]
        print(len(df_pointing))

        self.df = df_pointing.merge(self.df, how = 'left', left_on = 'obs_date', right_on = 'date')

        if target == 'ACTUALAZ_MEDIAN':
            self.df[target] = self.df[target] - self.df['ca']
        elif target == 'ACTUALEL_MEDIAN':
            self.df[target] = self.df[target] - self.df['ie']
        
        self.df = self.df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN', target]]
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()

        X = self.df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']]
        y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        self.target = [target]
        self.n_targets = 1

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
        


class PrepareData():

    def __init__(self,
                df_path              = './Data/processed_v4/all_features_safe.csv',
                params               = dataset_params
                ) -> None:

        targets = {
            "total": ["Offset"],
            "az"   : ["Off_Az_new"],
            "el"   : ["Off_El_new"],
            "both" : ["Off_El_new", "Off_Az_new"],
            'eel'  : ['Off_eEl'],
            'eaz'  : ['Off_eAz'],
            'real_az' : ['REALAZ'],
            'real_el' : ['REALEL'],
            }

        self.params = params
        self.scaled = False
        self.target = targets[params['target']]
        self.n_targets = len(self.target)
        self.df = pd.read_csv(df_path)

        if params['use_patches']:
            self.filter_patch(patches[params['patch_key']])

        if params['use_features']:
            if params['feature_key'] == 'new':
                self.df = self.df.loc[ : , ~self.df.columns.isin( features[ params['feature_key'] ] )]
            else:
                self.df = self.df.loc[ : , self.df.columns.isin( features[ params['feature_key'] ] )]


        if params['use_cartesian']:
            self.use_cartesian()

        df_pointing = pd.read_csv('./Data/tmp2022_clean_v2.csv') # df with offsets
        df_pointing.insert(0, 'Offset', np.sqrt(df_pointing['Off_El_new']**2 + df_pointing['Off_Az_new']**2))

        df_pointing['HOURS_SINCE_CORRECTION'] = (pd.to_datetime(df_pointing['obs_date']) - pd.to_datetime(df_pointing['timeLastCorrection'])) / pd.Timedelta('1h')
        self.instruments = list(df_pointing['rx'].unique())
        

        self.df = self.df.merge(df_pointing.loc[: , ['obs_date', 'ca_new', 'ie_new', 'rx', 'HOURS_SINCE_CORRECTION'] + self.target], how = 'left', left_on='date', right_on='obs_date')
        
        if params['filter_instruments']:
            self.df = self.df[self.df.rx == params['rx']]
            self.df = self.df.loc[: , self.df.columns != 'rx']
        else:
            dummies = pd.get_dummies(self.df['rx'])
            self.df = pd.concat([self.df.loc[: , self.df.columns != 'rx'], dummies], axis = 1)

        self.df = self.df.loc[ : , self.df.columns != 'obs_date']
        self.df = self.df.drop_duplicates(keep = 'first')
        self.df = self.df.dropna()
        self.transform_features()
        #Removes actualaz and actualel, and replaces them with cartesian coordinate
        
        # embed()
        polluted = False
        if polluted:
            #self.df.insert(0, 'polluted_az', self.df['Off_Az'])
            self.df.insert(0, 'polluted_el', self.df['Off_El_new'])

        if params['remove_outliers']:
            self.remove_outliers()
    

        print(self.df.columns)
        self.train_test_split_days()

        X_train, y_train = self.split_df(self.df_train, target = self.target)
        X_test, y_test   = self.split_df(self.df_test , target = self.target)

        if params['use_scaler']:
            X_train, X_test, y_train, y_test = self.scale_data(X_train, X_test, y_train, y_test, scaler = params['scaler'])

        if params['use_pca']:
            X_train, X_test = self.PCA(X_train, X_test, n_components = params['pca_components'])

        #Turn x and y into attributes
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


    def transform_features(self):
        #Transform all negative values of El_sun to 0
        self.df.loc[self.df['SUNEL_MEDIAN'] < 0, ['SUNAZ_MEDIAN','SUNEL_MEDIAN']] = 0
        #Hours since last scan


    def PCA(self, X_train, X_test, n_components, inverse_transform = False):
        #sklearn implementation of PCA
        if inverse_transform is True:
            X_train = self.pca.inverse_transform(X_train)
            X_test  = self.pca.inverse_transform(X_test)
            
        else:
            print('Before PCA:',X_train.shape)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X_train)
            X_train = self.pca.transform(X_train)
            X_test  = self.pca.transform(X_test)
            print('After PCA:', X_train.shape)
    
        return X_train, X_test

    def use_cartesian(self):
        """
        Transforms data with az and el to cartesian coordinates
        """

        df = self.df

        if 'ACTUALAZ' in df.columns and 'ACTUALEL' in df.columns:
            df['ACTUALAZ'] = np.deg2rad(df['ACTUALAZ'])
            df['ACTUALEL'] = np.deg2rad(df['ACTUALEL'])

            x = np.sin(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            y = np.cos(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
            z = np.sin(df['ACTUALEL'].values)

            df.insert(0, 'X', x)
            df.insert(0, 'Y', y)
            df.insert(0, 'Z', z)        
        
        if 'Az_sun' in df.columns and 'El_sun' in df.columns:            
            df['Az_sun'] = np.deg2rad(df['Az_sun'])
            df['El_sun'] = np.deg2rad(df['El_sun'])

            x_sun = np.sin(df['Az_sun'].values) * np.cos(df['El_sun'].values)
            y_sun = np.cos(df['Az_sun'].values) * np.cos(df['El_sun'].values)
            z_sun = np.sin(df['El_sun'].values)

            df.insert(0, 'X_sun', x_sun)
            df.insert(0, 'Y_sun', y_sun)
            df.insert(0, 'Z_sun', z_sun)


        if 'SunAzDiff' in df.columns and 'SunElDiff' in df.columns:
            df['SunAzDiff'] = np.deg2rad(df['SunAzDiff'])
            df['SunElDiff'] = np.deg2rad(df['SunElDiff'])

            x_sun_diff = np.sin(df['SunAzDiff'].values) * np.cos(df['SunElDiff'].values)
            y_sun_diff = np.cos(df['SunAzDiff'].values) * np.cos(df['SunElDiff'].values)
            z_sun_diff = np.sin(df['SunElDiff'].values)

            df.insert(0, 'X_sun_diff', x_sun_diff)
            df.insert(0, 'Y_sun_diff', y_sun_diff)
            df.insert(0, 'Z_sun_diff', z_sun_diff)


        if 'WINDDIR DIFF' in df.columns:
            df['WINDDIR DIFF'] = np.deg2rad(df['WindDirDiff'])

            x_wind = np.sin(df['WINDDIR DIFF'].values)
            y_wind = np.cos(df['WINDDIR DIFF'].values)
        
            df.insert(0, 'X_wind', x_wind)
            df.insert(0, 'Y_wind', y_wind)
        
        old_cols = ['ACTUALAZ', 'ACTUALEL', 'Az_sun', 'El_sun', 'SunAzDiff', 'SunElDiff', 'WINDDIR DIFF']
        df = df.loc[ : , ~df.columns.isin(old_cols)]
        
        self.df = df


    def filter_patch(self, patch: tuple, rotation = 23):
        """
        Filters self.df to only include data from a patch
        - If len(patch) is 4 -> Filters from left right top bottom with az and el
        - If len(patch) is 2 -> Transform into cartesian coordinates, rotate around 
          x-axis such that the lines are perpendicular to y-axis, then filter between the two y-values.
        
        """
        df = self.df

        if len(patch) == 4:
            l,r,t,b = np.deg2rad(np.array(patch))
            
            df.insert(0, 'ACTUALAZ CUT', df['ACTUALAZ_MEDIAN'])
            df.loc[df['ACTUALAZ CUT'] >  np.pi, 'ACTUALAZ CUT'] -= 2*np.pi
            df.loc[df['ACTUALAZ CUT'] < -np.pi, 'ACTUALAZ CUT'] += 2*np.pi
            df = df.loc[ (df['ACTUALAZ CUT'] > l) & (df['ACTUALAZ CUT'] < r) & (df['ACTUALEL_MEDIAN'] > b) & (df['ACTUALEL_MEDIAN'] < t) ]
            df = df.loc[ : , df.columns != 'ACTUALAZ CUT']

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

        self.df = df
        return

    def train_test_split_days(self):

        df = self.df
        df['date']  = pd.to_datetime(df['date'])
        df.insert(0, 'day', df['date'].dt.date)
        dfs = [df[df['day'] == day] for day in df['day'].unique()]
        random.Random(random_seed).shuffle(dfs)

        train_size = 0.78
        n_days     = len(dfs)

        dfs_train = dfs[:int(train_size * n_days)]
        dfs_test  = dfs[int(train_size * n_days):]

        self.df_train = pd.concat(dfs_train)
        self.df_test  = pd.concat(dfs_test)

        self.df_train = self.df_train.loc[: , self.df_train.columns != 'day']
        self.df_test  = self.df_test.loc [: , self.df_test.columns  != 'day']
        train_days = len(self.df_train)
        test_days  = len(self.df_test)
        print(f'Training days: {train_days} | Test days: {test_days} | Train size: {train_days/(train_days+test_days):.2f}')
        return

    def remove_outliers(self, remove_from_target = True):
        non_val_cols = ['Hour', 'date']
        factor = self.params['outlier_threshold']

        if remove_from_target:
            non_val_cols = ['Off_El_new', 'Off_Az_new', 'Off_eEl', 'Off_eAz'] # one or more

        else:
            non_val_cols = list(self.df.loc[: , ~self.df.columns.isin(['Off_Az_new', 'Off_El_new', 'Hour', 'date', 'SunAboveHorizon','ie','ca','TEMP1','TEMP27','TILT1T'] + self.instruments)].columns)

        Q1 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.25)
        Q3 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.75)
        IQR = Q3 - Q1

        ## Will raise ValueError in the future
        print('Length before removing outliers:', len(self.df))
        self.df = self.df[~((self.df.loc[: , self.df.columns.isin(non_val_cols)] < (Q1 - factor * IQR)) |(self.df.loc[: , self.df.columns.isin(non_val_cols)] > (Q3 + factor * IQR))).any(axis=1)]
        print('Length after removing outliers:', len(self.df))
        return

    def split_df(self, df, target):
        X = df.loc[:, ~ df.columns.isin( ['date'] + target )]
        self.xcols = X.columns
        y = df.loc[:, target]
        return X, y


    def scale_data(self, X_train, X_test, y_train, y_test, scaler = 'StandardScaler'):
        print(f"Scaling data with {scaler}")
        
        scaler_dict = {'StandardScaler': StandardScaler(), 'PowerTransformer': PowerTransformer()}

        self.scaler1 = scaler_dict[scaler]
        self.scaler2 = scaler_dict[scaler]

        X_train = self.scaler1.fit_transform(X_train.values)
        X_test = self.scaler1.transform(X_test.values)

        if self.n_targets > 1:
            y_train = self.scaler2.fit_transform(y_train.values)
            y_test = self.scaler2.transform(y_test.values)

        else:
            y_train = self.scaler2.fit_transform(y_train.values.reshape(-1,1)).ravel()
            y_test = self.scaler2.transform(y_test.values.reshape(-1,1)).ravel()

        self.scaled = True

        return X_train, X_test, y_train, y_test


    def rescale_data(self, y):
        print('Rescaling data for evaluation')

        if self.n_targets > 1:
            y = self.scaler.inverse_transform(y)
            
        else:
            y = self.scaler.inverse_transform(y.reshape(-1,1)).ravel()

        self.scaled = False
        
        return y


class PointingScansClassification():

    def __init__(self, PATH_SCANS, var = 'az', use_upsampling = True, use_scaler = False, use_rx = False):
        """
        Prepares data for ML models.
        This class is for the pointing scan classification
        
        parameters
        PATH_SCANS: path to .csv file contianing pointing scans 
        var: either az or el.
        """

        self.var = var
        otherVar = self.test_var(var)
        

        """        #Find the good and bad scans
        dir_good = os.listdir(f'./PointingScanPlots/good_{var}')
        dir_bad = os.listdir(f'./PointingScanPlots/good_{otherVar}')
        dir_good_both = os.listdir('./PointingScanPlots/good_both')
        dir_bad_both = os.listdir('./PointingScanPlots/bad_both')

        good_both = [filename.split('_')[2] for filename in dir_good_both]
        bad_both  = [filename.split('_')[2] for filename in dir_bad_both]

        good  = [filename.split('_')[2] for filename in dir_good] + good_both
        bad   = [filename.split('_')[2] for filename in dir_bad] + bad_both
        """

        dir_good = os.listdir(f'./PointingScanPlots_v2/Good/')
        dir_good = os.listdir(f'./PointingScanPlots_v2/Bad/')
        
        good  = [filename.split('_')[2] for filename in dir_good] 
        bad   = [filename.split('_')[2] for filename in dir_bad] 

        scans = np.array(good + bad).astype(int)
        good = np.array(good).astype(int)

    
        df = pd.read_csv(PATH_SCANS)
        df['obs_date'] = pd.to_datetime(df['obs_date'])
        df['beamsize'] = 7.8 * 800 / df['freq']
        df = df.sort_values(by='obs_date')
        df = clean_pointing_table(df)

        df = df.loc[df.scan.isin(scans)]
        dummies_rx = pd.get_dummies(df['rx'])

        # df[f'Amp_rAz'] = df.Amp_Az / df.Amp_eAz
        # df[f'Amp_rEl'] = df.Amp_El / df.Amp_eEl
        # df[f'FWHM_rAz'] = df.FWHM_Az / df.FWHM_eAz
        # df[f'FWHM_rEl'] = df.FWHM_El / df.FWHM_eEl
        # df[f'Off_rAz'] = df.Off_Az / df.Off_eAz
        # df[f'Off_rEl'] = df.Off_El / df.Off_eEl

        features = ['scan','beamsize','rx',
            f'Off_{var.capitalize()}',
            f'Off_e{var.capitalize()}',
            f'FWHM_{var.capitalize()}',
            f'FWHM_e{var.capitalize()}',
            f'Amp_{var.capitalize()}',
            f'Amp_e{var.capitalize()}'
            ]
        self.n_features = len(features)
        

        df = df[features]
        df['good'] = np.where(df.scan.isin(good), 1, 0)

        if 'rx' in features:
            le = LabelEncoder()
            df['rx'] = le.fit_transform(df['rx'])

        if use_rx:
            df = pd.concat([df, dummies_rx], axis = 1)
    
        # split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'good'], df['good'], test_size=0.2, random_state=0)

        if use_upsampling:
            X_train, y_train = self.upsample(X_train, y_train)


        if use_scaler:
            X_train, X_test, self.scale(X_train, X_test)


        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if type(self.X_train) != np.ndarray:
            self.X_train = self.X_train.values
            self.y_train = self.y_train.values

        if type(self.X_test) != np.ndarray:
            self.X_test = self.X_test.values
            self.y_test = self.y_test.values

        print(type(self.X_train), type(self.y_train), type(self.X_test), type(self.y_test))
    
    def get_data(self):
        return self.X_train[:,1:], self.X_test[:,1:], self.y_train, self.y_test

    def add_rx(self):
        pass
    
    def test_var(self,var):
        """Test if var is valid and return the other variable"""
        if var == 'az':
            return 'el'
        elif var == 'el':
            return 'az'
        else:
            raise ValueError ('var must be az or el')

    def scale(self, X_train, X_test):
        scaler = StandardScaler()
        # Fit the transformers on the training df and transform both training and test df
        X_train[:,1:self.n_features] = scaler.fit_transform(X_train[:,1:self.n_features])
        X_test[:,1:self.n_features] = scaler.transform(X_test[:,1:self.n_features])

        return X_train, X_test

    def upsample(self, X_train, y_train):
        # combine the training data and split into good and bad scans
        data_train = pd.concat([X_train, y_train], axis=1)
        good_scans_train = data_train[data_train.good == True]
        bad_scans_train = data_train[data_train.good == False]

        # upsample the bad scans in the training data to match the number of good scans
        bad_scans_upsampled = resample(bad_scans_train, replace=True, n_samples=len(good_scans_train))

        # combine the upsampled bad scans with the good scans to form the new training data
        data_train_upsampled = pd.concat([good_scans_train, bad_scans_upsampled])

        # extract the features and labels from the upsampled training data
        X_train_upsampled = data_train_upsampled.iloc[:, :-1].values
        y_train_upsampled = data_train_upsampled['good'].values

        X_train = X_train_upsampled
        y_train = y_train_upsampled
        
        return X_train, y_train



class PointingScansClassification_v2():

    def __init__(self, PATH_SCANS, use_upsampling = False, use_scaler = False, use_rx = False):
        """
        Prepares data for ML models.
        This class is for the pointing scan classification
        
        parameters
        PATH_SCANS: path to .csv file contianing pointing scans 
        var: either az or el.
        """

        self.var = 'both'

        #Find the good and bad scans
        dir_good = os.listdir(f'./PointingScanPlots_v2/Good/')
        dir_bad = os.listdir(f'./PointingScanPlots_v2/Bad/')
        
        good  = [filename.split('_')[2] for filename in dir_good] 
        bad   = [filename.split('_')[2] for filename in dir_bad] 

        scans = np.array(good + bad).astype(int)
        good = np.array(good).astype(int)

    
        df = pd.read_csv(PATH_SCANS)
        df['obs_date'] = pd.to_datetime(df['obs_date'])
        df['beamsize'] = 7.8 * 800 / df['freq']
        df = df.sort_values(by='obs_date')
        df = clean_pointing_table(df)

        df = df.loc[df.scan.isin(scans)]
        dummies_rx = pd.get_dummies(df['rx'])

        # df[f'Amp_rAz'] = df.Amp_Az / df.Amp_eAz
        # df[f'Amp_rEl'] = df.Amp_El / df.Amp_eEl
        # df[f'FWHM_rAz'] = df.FWHM_Az / df.FWHM_eAz
        # df[f'FWHM_rEl'] = df.FWHM_El / df.FWHM_eEl
        # df[f'Off_rAz'] = df.Off_Az / df.Off_eAz
        # df[f'Off_rEl'] = df.Off_El / df.Off_eEl

        features = ['scan',
                'Off_Az', 'Off_eAz', 'FWHM_Az', 'FWHM_eAz', 'Amp_Az', 'Amp_eAz',
                'Off_El', 'Off_eEl', 'FWHM_El', 'FWHM_eEl', 'Amp_El', 'Amp_eEl'
                ]

        self.n_features = len(features)
        self.feature_names = features[1:]

        df = df[features]
        df['good'] = np.where(df.scan.isin(good), 1, 0)

        if 'rx' in features:
            le = LabelEncoder()
            df['rx'] = le.fit_transform(df['rx'])

        if use_rx:
            df = pd.concat([df, dummies_rx], axis = 1)
    
        # split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'good'], df['good'], test_size=0.20, random_state=0)

        if use_upsampling:
            X_train, y_train = self.upsample(X_train, y_train)


        if use_scaler:
            X_train, X_test, self.scale(X_train, X_test)


        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if type(self.X_train) != np.ndarray:
            self.X_train = self.X_train.values
            self.y_train = self.y_train.values

        if type(self.X_test) != np.ndarray:
            self.X_test = self.X_test.values
            self.y_test = self.y_test.values

        print(type(self.X_train), type(self.y_train), type(self.X_test), type(self.y_test))
    
    def get_data(self):
        return self.X_train[:,1:], self.X_test[:,1:], self.y_train, self.y_test

    def get_neg_over_pos(self):
        # Returns the ratio of positive samples to negative samples
        return np.sum(self.y_train == 0) / np.sum(self.y_train == 1)

    def add_rx(self):
        pass
    
    def test_var(self,var):
        """Test if var is valid and return the other variable"""
        if var == 'az':
            return 'el'
        elif var == 'el':
            return 'az'
        else:
            raise ValueError ('var must be az or el')

    def scale(self, X_train, X_test):
        scaler = StandardScaler()
        # Fit the transformers on the training df and transform both training and test df
        X_train[:,1:self.n_features] = scaler.fit_transform(X_train[:,1:self.n_features])
        X_test[:,1:self.n_features] = scaler.transform(X_test[:,1:self.n_features])

        return X_train, X_test

    def upsample(self, X_train, y_train):
        # combine the training data and split into good and bad scans
        data_train = pd.concat([X_train, y_train], axis=1)
        good_scans_train = data_train[data_train.good == True]
        bad_scans_train = data_train[data_train.good == False]

        # upsample the bad scans in the training data to match the number of good scans
        bad_scans_upsampled = resample(bad_scans_train, replace=True, n_samples=len(good_scans_train))

        # combine the upsampled bad scans with the good scans to form the new training data
        data_train_upsampled = pd.concat([good_scans_train, bad_scans_upsampled])

        # extract the features and labels from the upsampled training data
        X_train_upsampled = data_train_upsampled.iloc[:, :-1].values
        y_train_upsampled = data_train_upsampled['good'].values

        X_train = X_train_upsampled
        y_train = y_train_upsampled
        
        return X_train, y_train


if __name__ == '__main__':

    test = PrepareDataAnalytical_v3()

