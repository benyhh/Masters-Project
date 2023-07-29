import importlib
import functions
importlib.reload(functions)
from functions import *
from functions import random_seed

random.seed(random_seed)
np.random.seed(random_seed)

#paths 
path_data = "./Data/"
path_correlations = "./Correlations/"
path_plots = "./Plots/"
path_features = "./Features/"
path_models = "./Models/"
path_pointing = os.path.join(path_data,"./PointingTable.csv")
path_meteoscope = os.path.join(path_data, "./meteoscope.csv")
path_merged = os.path.join(path_data, "DF2_merged.csv")
#createMerged(writeFile = True)

#df = pd.read_csv(os.path.join(path_data, "DF2_merged.csv"))
#createMerged(writeFile = True)
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("seaborn")
sns.set(font_scale=1.5)
plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18
#plt.rcParams["font.family"] = "Times New Roman"

patches = {
    0: 	None,
    1:	(-75,75,75,15),
    # 2:	(-28,28,76,60),
    3:	(-120,-98,63,16),
    # 4:	(-75,0,75,15),
    # 5: 	(-28,28,57,47),
    # 5: (-0.083, 0.057),
    # 6: (0.195, 0.266),
    7:  (0,90,90,0),
    8:  (90,180,90,0),
    9:  (-180,-90,90,0),
    10: (-90,0,90,0),
    # 11: (-180,180,75,15),
    # 12: (-180,180,75,45),
    # 13: (-180,180,45,15),

    } #


feature_lists = {
    # 'all':    ['ACTUALAZ', 'ACTUALEL', 'TEMP1', 'TEMP26', 'TEMP28', 'TILT1T', 'Az_sun', 'El_sun', 'SunElDiff',
    #            'SunAzDiff', 'SunAngleDiff', 'SunAngleDiff_15', 'POSITIONX', 'POSITIONY', 'PRESSURE', 'HUMIDITY',
    #            'WINDDIR DIFF', 'TURBULENCE', 'Hour', 'date'],

    # # 'offset': ['TEMP1', 'TEMP26', 'TEMP28', 'TILT1T', 'Az_sun','SunAzDiff', 'POSITIONY', 'PRESSURE', 'Hour', 'date'],

    # 'sf2':    ['ACTUALAZ', 'ACTUALEL', 'TEMP1', 'TILT1T', 'SunAngleDiff', 'SunAngleDiff_15', 'POSITIONY', 'HUMIDITY',
    #            'TURBULENCE', 'Hour', 'date'],

    #'None':   None,

    'Corr':   ['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
               'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
               'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
               'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','ROTATIONX_sumdabs2','ACTUALAZ_sumdabs2',
               'TILT1X_sumdabs2','ACTUALEL_sumdabs5','POSITIONX_sumdabs5','ROTATIONX_sumdabs5',
               'DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT', 'DAZ_TOTAL', 'DEL_TOTAL', 'date'],

    'Corr_reduced':['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
                    'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
                    'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                    'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1',
                    'DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT','date'],

    'Corr_reduced2':['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TILT1X','WINDDIRECTION',
                    'SunAngleDiff','SunAngleDiff_15','SunElDiff','Hour',
                    'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                    'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT','date'],
    
    'Corr_reduced3':['ACTUALAZ','ACTUALEL','TEMP1','TILT1X','WINDDIRECTION','SunAngleDiff_15','Hour',
                    'TURBULENCE','WINDDIR DIFF','DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT', 'date'],

    'hp_el1':          ['DEL_TILT', 'WINDDIRECTION', 'POSITIONZ', 'ACTUALAZ', 'HUMIDITY', 'SunAngleDiff_15', 'TILT1X_sumdabs1', 'WINDDIR DIFF', 'TURBULENCE', 'Hour', 'date'],            
    
    'hp_az0':          ['SunElDiff', 'DEL_TILT', 'DAZ_TILT', 'DAZ_TILTTEMP', 'TILT1X_sumdabs1', 'ACTUALEL', 'WINDDIRECTION',
                        'DAZ_DISP', 'SunElDiff', 'ACTUALEL', 'TILT1X', 'TURBULENCE', 'ROTATIONX_sumdabs1', 'TEMP1', 'WINDDIR DIFF','date']
    }



class MeanAbsoluteError(Loss):
    def call(self, y_true, y_pred):
        x = tf.math.subtract(y_true, y_pred)
        x = tf.math.square(x)
        x = tf.math.reduce_sum(x, axis = 1)
        x = tf.math.reduce_mean(x)
        return x

class MAE_loss():
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.mean(np.linalg.norm(y_true-y_pred, axis = 1)**2)


class Model:
    def __init__(self,
                df_path              = '/Data/merged_features_all.csv',
                method               = "XGBoost",
                target               = "total",
                load_model           = None,
                selected_columns_key = None,
                patch_key            = None,
                use_pca              = True
                ) -> None:
        
        models = {
            "XGBoost": self.XGBoost,
            "NN": self.NN,
            'NNKeras': self.NNKeras,
            "RandomForest": self.RandomForest
            }

        model_files = {
            "XGBoost": "model_xgb",
            "NN": "model_nn",
            "RandomForest": "model_rf"
            }

        targets = {
            "total": ["Offset"],
            "az"   : ["Off_Az"],
            "el"   : ["Off_El"],
            "both" : ["Off_El", "Off_Az"]
            }
        
        self.n_targets = len(targets[target])
        self.df = pd.read_csv(df_path)

        patch            = patches[patch_key]
        selected_columns = feature_lists[selected_columns_key]

        polluted      = False
        self.polluted = polluted

        if patch is not None:
            self.filter_patch(patch)
    
        if selected_columns is not None:
            self.df = self.df.loc[ : , self.df.columns.isin(selected_columns)]
        
        self.selected_columns_key = selected_columns_key
        self.patch_key            = patch_key

        self.model_function = models[method]
        self.name           = method
        self.target         = target
        self.targets        = targets
        self.scaled         = False
        self.PCA_transform  = use_pca
        use_instruments     = True

        self.PATH_MODEL = f'./Results/{patch_key}/{selected_columns_key}/Models/' 
        self.PATH_PLOTS = f'./Results/{patch_key}/{selected_columns_key}/Plots/'
        if self.PCA_transform:
            self.FN_MODEL = os.path.join(self.PATH_MODEL, f'{self.name}_pca_{self.target}.sav')
            self.FN_PCA   = os.path.join(self.PATH_MODEL, f'{self.name}_pcatransform_{self.target}.pkl')
        else:
            self.FN_MODEL = os.path.join(self.PATH_MODEL, f'{self.name}_{self.target}.sav')
            
        print('-----------------')
        print(f"Model: {self.name} | Target: {self.target} | Features: {selected_columns_key} | Patch: {patch_key}")


        if not os.path.exists(self.PATH_MODEL):
            os.makedirs(self.PATH_MODEL)
        if not os.path.exists(self.PATH_PLOTS):
            os.makedirs(self.PATH_PLOTS)
        
        
        df_pointing = pd.read_csv('./Data/PointingTable.csv') # df with offsets
        df_pointing.insert(0, 'Offset', np.sqrt(df_pointing['Off_El']**2 + df_pointing['Off_Az']**2))
        
        if use_instruments is True:
            self.instruments = list(df_pointing['rx'].unique())
            self.df = self.df.merge(df_pointing.loc[: , ['obs_date', 'ca', 'ie', 'rx'] + targets[target]], how = 'left', left_on='date', right_on='obs_date')
            dummies = pd.get_dummies(self.df['rx'])
            self.df = pd.concat([self.df.loc[: , self.df.columns != 'rx'], dummies], axis = 1)


        self.df = self.df.loc[ : , self.df.columns != 'obs_date']
        self.df = self.df.drop_duplicates(subset = ['date'], keep = 'first')

        if polluted:
            self.df.insert(0, 'polluted_az', self.df['Off_Az'])
            self.df.insert(0, 'polluted_el', self.df['Off_El'])

        #pd.set_option('display.max_columns', None)
        self.remove_outliers()
        #self.remove_outliers(from_target=False)

        self.train_test_split_days()
        self.X_train, self.y_train = self.split_df(self.df_train, target = targets[target])
        self.X_test, self.y_test   = self.split_df(self.df_test , target = targets[target])

        self.scale_data()

        if load_model is True:
            if os.path.exists(self.FN_MODEL):
                print("Loading pretrained model from", self.FN_MODEL)
                self.model  = pickle.load(open(self.FN_MODEL, 'rb'))
                self.model  = self.model
                self.params = self.model.get_params()
                if self.PCA_transform:
                    self.pca = pickle.load(open(self.FN_PCA, 'rb'))
                    print(load_model)
            else:
                print("No pretrained model found in", self.FN_MODEL)
    

        if self.PCA_transform:
            self.X_train, self.X_test = self.PCA(self.X_train, self.X_test, loaded = load_model)


            n_pcs= self.pca.components_.shape[0]

            most_important = [np.argpartition(self.pca.components_[i], -3)[-3:] for i in range(n_pcs)]

            initial_feature_names = self.df.loc[: , ~self.df.columns.isin( ['date'] + targets[target])].columns
            # get the names
            most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

            # LIST COMPREHENSION HERE AGAIN
            dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}


            # build the dataframe
            df_pca = pd.DataFrame(dic.items())
            self.xcols = list(df_pca[1].values)

        
        #convert x and y train and test to numpy if they are not already
        if not isinstance(self.X_train, np.ndarray):
            self.X_train = self.X_train.values
        if not isinstance(self.X_test, np.ndarray):
            self.X_test = self.X_test.values
        if not isinstance(self.y_train, np.ndarray):
            self.y_train = self.y_train.values.ravel()
        if not isinstance(self.y_test, np.ndarray):
            self.y_test = self.y_test.values.ravel()
    
        # print(f'X_train | NaN: {np.sum(np.isnan(self.X_train))} | Min: {np.min(self.X_train)} | Max: {np.max(self.X_train)} |')
        # print(f'X_test | NaN: {np.sum(np.isnan(self.X_test))} | Min: {np.min(self.X_test)} | Max: {np.max(self.X_test)} |')
        # print(f'y_train | NaN: {np.sum(np.isnan(self.y_train))} | Min: {np.min(self.y_train)} | Max: {np.max(self.y_train)} |')
        # print(f'y_test | NaN: {np.sum(np.isnan(self.y_test))} | Min: {np.min(self.y_test)} | Max: {np.max(self.y_test)} |')

    def PCA(self, X_train, X_test, loaded = False, inverse_transform = False):
        #sklearn implementation of PCA
        if inverse_transform is True:
            X_train = self.pca.inverse_transform(X_train)
            X_test  = self.pca.inverse_transform(X_test)
            
        else:
            print('Before PCA:',X_train.shape)
            if not loaded:
                print('In if not loaded')
                self.pca = PCA(n_components=0.99)
                self.pca.fit(X_train)


            X_train = self.pca.transform(X_train)
            X_test  = self.pca.transform(X_test)

            print('After PCA:',X_train.shape)
    
        return X_train, X_test

        
    def filter_patch(self, patch: tuple, rotation = 23):
        """
        Filters self.df to only include data from a patch
        - If len(patch) is 4 -> Filters from left right top bottom with az and el
        - If len(patch) is 2 -> Transform into cartesian coordinates, rotate around 
          x-axis such that the lines are perpendicular to y-axis, then filter between the two y-values.
        
        """
        df = self.df

        if len(patch) == 4:
            l,r,t,b = patch
            df.insert(0, 'ACTUALAZ CUT', df['ACTUALAZ'])
            df.loc[df['ACTUALAZ CUT'] >  180, 'ACTUALAZ CUT'] -= 360
            df.loc[df['ACTUALAZ CUT'] < -180, 'ACTUALAZ CUT'] += 360
            df = df.loc[ (df['ACTUALAZ CUT'] > l) & (df['ACTUALAZ CUT'] < r) & (df['ACTUALEL'] > b) & (df['ACTUALEL'] < t) ]
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

        train_size = 0.75
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


    def remove_outliers(self, from_target = True):
        non_val_cols = ['Hour', 'date']
        if from_target:
            factor = 1.7
            non_val_cols = ['Off_El', 'Off_Az'] # one or more

        else:
            factor = 1.7
            non_val_cols = list(self.df.loc[: , ~self.df.columns.isin(['Off_Az', 'Off_El', 'Hour', 'date', 'SunAboveHorizon','ie','ca','TEMP1','TEMP27','TILT1T'] + self.instruments)].columns)

        Q1 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.25)
        Q3 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.75)
        IQR = Q3 - Q1

        ## Will raise ValueError in the future
        self.df = self.df[~((self.df.loc[: , self.df.columns.isin(non_val_cols)] < (Q1 - factor * IQR)) |(self.df.loc[: , self.df.columns.isin(non_val_cols)] > (Q3 + factor * IQR))).any(axis=1)]

   

    def scale_data(self, scaler = 'StandardScaler'):
        print("Scaling data")
        
        scaler_dict = {'StandardScaler': StandardScaler(), 'PowerTransformer': PowerTransformer()}

        self.scaler1 = scaler_dict[scaler]
        self.scaler2 = scaler_dict[scaler]
        
        self.X_train = self.scaler1.fit_transform(self.X_train.values)
        self.X_test = self.scaler1.transform(self.X_test.values)

        if self.n_targets > 1:
            self.y_train = self.scaler2.fit_transform(self.y_train.values)
            self.y_test = self.scaler2.transform(self.y_test.values)
            
        else:
            self.y_train = self.scaler2.fit_transform(self.y_train.values.reshape(-1,1)).ravel()
            self.y_test = self.scaler2.transform(self.y_test.values.reshape(-1,1)).ravel()
        
        self.scaled = True
    
    def rescale_data(self):
        print('Rescaling data for evaluation')
        if self.n_targets > 1:
            self.y_train = self.scaler2.inverse_transform(self.y_train)
            self.y_test = self.scaler2.inverse_transform(self.y_test)
            
        else:
            self.y_train = self.scaler2.inverse_transform(self.y_train.reshape(-1,1)).ravel()
            self.y_test = self.scaler2.inverse_transform(self.y_test.reshape(-1,1)).ravel()

        self.scaled = False

    def split_df(self, df, target):
        X = df.loc[:, ~ df.columns.isin( ['date'] + target )]
        self.xcols = X.columns
        y = df.loc[:, target]
        return X, y


    def train(self, save = True):
        self.model_function(save)

    def evaluate(self, verbose = True):

        pred = self.model.predict(self.X_test)

        # Rescaling data if necessary
        if self.scaled:
            self.rescale_data()

        
        if self.n_targets > 1:
            for i,t in enumerate(self.targets[self.target]):
                target = t.split('_')[1]
                me = mean_absolute_error(pred[:,i], self.y_test[:,i])
                if verbose:
                    print(f'Mean error for {target}: {me:.3f}')
        
            me = np.mean(np.linalg.norm(self.y_test-pred, axis = 1))
        else:
            me = mean_absolute_error(pred, self.y_test)
        
        if verbose:
            print(f'Mean error for {self.target}: {me:.3f}')

        return me

    def plot_sorted_predictions(self):
        print(f"Plotting sorted predictions for {self.name}")

        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)
        if not os.path.exists(PATH_HISTOGRAM):
            os.makedirs(PATH_HISTOGRAM)
        
        me = self.evaluate(verbose = False)
        pred = self.model.predict(self.X_test)
        if self.target == 'both':
            noPred = np.mean(np.sqrt(self.df['Off_Az']**2 + self.df['Off_El']**2))
        else:
            noPred = self.df.loc[:, self.targets[self.target]].abs().values.mean() 


        plt.clf()
        plt.figure(figsize=(18,12))


        if self.n_targets > 1:
            idxSorted0 = self.y_test[:,0].argsort()
            idxSorted1 = self.y_test[:,1].argsort()
        
            plt.plot(range(len(pred)), pred[idxSorted0,0], label="Prediction Az")
            plt.plot(range(len(pred)), pred[idxSorted1,1], label="Prediction El")
            
            
            plt.plot(range(len(pred)), self.y_test[idxSorted0, 0], label='Real Az')
            plt.plot(range(len(pred)), self.y_test[idxSorted1, 1], label='Real El')

        else:
            idxSorted = self.y_test.argsort()
            plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
            
            try:
                plt.plot(range(len(pred)), self.y_test[idxSorted], label="Real data")
            except:
                idxSorted = self.y_test.ravel().argsort()
                plt.plot(range(len(pred)), self.y_test.ravel()[idxSorted], label="Real data")
        
        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | ME: {me:.3f} | {noPred:.3f}")
        plt.legend()


        if self.PCA_transform is True:
            save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_pca_{self.target}.png")
            save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_pca_{self.target}.png")
        else:
            save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_{self.target}.png")
            save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_{self.target}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()

        titlesize = 32
        fontsize  = 26

        n_bins = 25
        plt.figure(figsize=(18,12))
        _, bins, _ = plt.hist(self.y_test, bins = n_bins, alpha = 0.8, label = 'Current offset')
        plt.hist(self.y_test - pred, bins = bins, alpha = 0.8, label='Offset with ML model applied')
        plt.xlabel('Offset [\'\']', fontsize = fontsize)
        plt.ylabel('Number of samples', fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.title('Distribution of pointing offsets with and without ML model', fontsize = 30)
        plt.legend(fontsize = fontsize)
        plt.tight_layout()
        plt.savefig(save_path_hist, dpi = 800)

        return

    def plot_sorted_train(self):
        print(f'Plotting Sorted Predictions on Train set for {self.name}')

        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')

        pred = self.model.predict(self.X_train)
        idxSorted = self.y_train.argsort()

        ME = self.evaluate()

        plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
        try:
            plt.plot(range(len(pred)), self.y_train[idxSorted], label="Real data")
        except:
            idxSorted = self.y_train.ravel().argsort()
            plt.plot(range(len(pred)), self.y_train.ravel()[idxSorted], label="Real data")

        plt.xlabel("Observations")
        plt.ylabel("Normalized Offset")
        plt.title(f"{self.name} | ME: {ME:.3f}")
        plt.legend()
        plt.savefig(os.path.join(PATH_SORTEDPRED, f"sortpred_TRAIN_{self.name}_{self.target}.png"), dpi = 400)
        plt.clf()

        return
    
    def mean_error_loss(self, y_true, y_pred):
    
        x = tf.math

        print(y_pred)
        print(type(y_pred))
        if y_true.shape[1] > 1:
            me = np.mean(np.linalg.norm(y_true.numpy()-y_pred.numpy(), axis = 1))
        else:
            me = mean_absolute_error(y_true.numpy(), y_pred.numpy())
        
        return me

    def NNKeras(self, save = True):

        # model class to use in the scikit random search CV 
        model = KerasRegressor(build_fn=self.create_model, epochs=60, batch_size=64, verbose=0)
        
        # learning algorithm parameters
        lr=uniform(1e-4, 1e-1)
        decay= uniform(0, 1e-6)

        # activation
        activation=['relu', 'tanh']

        #callbacks
        # early_stopper = EarlyStopping(monitor='val_acc', patience=3, verbose=1)

        # numbers of layers
        # nl1 = randint(1,2)
        # nl2 = randint(0,2)
        # nl3 = randint(0,2)

        nl1 = [1]
        nl2 = [1]
        nl3 = [1]

        # neurons in each layer
        nn1=[150,200,300]
        nn2=[100,150,200]
        nn3=[150,200,300]

        # dropout and regularisation
        dropout = uniform(0,0.3)
        l1 = uniform(0.0001,0.1)
        l2 = uniform(0.0001,0.1)

        # dictionary summary
        param_grid = dict(
                            nl1=nl1, nl2=nl2, nl3=nl3, nn1=nn1, nn2=nn2, nn3=nn3,
                            act=activation, l1=l1, l2=l2, learning_rate=lr, decay=decay, dropout=dropout, 
                            input_shape=[self.X_train.shape[1]], output_shape = [self.n_targets],
                        )

        grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=param_grid, n_iter=100,
                                    n_jobs=-1, verbose = 1, return_train_score = True)
        
        grid_result = grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_
        self.params = self.model.get_params()
        cv_results = grid_result.cv_results_

        plt.clf()
        plt.plot(range(len(cv_results['mean_test_score'])), cv_results['mean_test_score'] , label = 'test')
        plt.plot(range(len(cv_results['mean_test_score'])), cv_results['mean_train_score'], label = 'train')
        plt.legend()
        plt.ylim(-1.5,0)
        plt.savefig(f'./Results/GridSearch/gridsearch_{self.name}_{self.patch_key}_{self.selected_columns_key}_{self.target}.png')
        
        df_results = pd.DataFrame(cv_results)
        df_results.to_csv(f'./Results/GridSearch/cv_results_{self.patch_key}_{self.selected_columns_key}_{self.target}.csv', index = False)
       
       

        if save:
            self.save_model()
    
    def create_model(self, nl1=1, nl2=1,  nl3=1, 
                 nn1=1000, nn2=500, nn3 = 200, learning_rate=0.01, decay=0., l1=0.01, l2=0.01,
                act = 'relu', dropout=0, input_shape=1000, output_shape=20):
        '''This is a model generating function so that we can search over neural net 
        parameters and architecture'''
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,  decay=decay)
        reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
        loss_function = MeanAbsoluteError()                          
        model = Sequential()
        
        # for the firt layer we need to specify the input dimensions
        first=True
        
        for i in range(nl1):
            if first:
                model.add(Dense(nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg))
                first=False
            else: 
                model.add(Dense(nn1, activation=act, kernel_regularizer=reg))
            if dropout!=0:
                model.add(Dropout(dropout))
                
        for i in range(nl2):
            if first:
                model.add(Dense(nn2, input_dim=input_shape, activation=act, kernel_regularizer=reg))
                first=False
            else: 
                model.add(Dense(nn2, activation=act, kernel_regularizer=reg))
            if dropout!=0:
                model.add(Dropout(dropout))
                
        for i in range(nl3):
            if first:
                model.add(Dense(nn3, input_dim=input_shape, activation=act, kernel_regularizer=reg))
                first=False
            else: 
                model.add(Dense(nn3, activation=act, kernel_regularizer=reg))
            if dropout!=0:
                model.add(Dropout(dropout))
                
        model.add(Dense(output_shape, activation=act))
        model.compile(loss=loss_function, optimizer=opt, metrics=['MeanSquaredError'])
        return model

    def NN(self, save = True):
        #train and hypertune sklearn neural network and save the best model
        print(f"Start NN training, number of features: {self.X_train.shape[1]}")
        
        hl2 = [(300,200,300),(250,160,100),(200,100,200), (200,160,200), (100,50,100), (100,100,100), (50,20,50), (100,100), (50,50), (100,)]

        parameters = {
            'hidden_layer_sizes':hl2,
            'activation':['relu','tanh'],
            'solver':['adam'],
            'learning_rate':['adaptive'],
            'max_iter':[2000],
            'alpha': uniform(0.0001, 0.01),
            # 'batch_size': randint(16,256),
            'early_stopping': [True],
            'learning_rate_init': uniform(0.0001,0.01),
            'beta_1': uniform(0.01,0.999-0.01),
            'beta_2': uniform(0.01,0.999-0.01),
            'n_iter_no_change': [20]
            }

        nn = MLPRegressor(random_state=random_seed)
        hl = [(20,40,20), (20,20,20), (100,50)] 

        loss_function = MAE_loss()

        scorer = make_scorer(loss_function, greater_is_better = False)

        self.model = RandomizedSearchCV(nn, parameters, cv=3, scoring=scorer , n_jobs = -1, n_iter = 500,
                                  verbose = 1, random_state = random_seed)
        # Fit the random search model
        if self.n_targets > 1:
            self.model = self.model.fit(self.X_train, self.y_train)
        else:
            self.model = self.model.fit(self.X_train, self.y_train.ravel())
    
        self.model = self.model.best_estimator_
        self.params = self.model.get_params()
        plt.plot(self.model.loss_curve_)
        plt.savefig('loss_curve.png', dpi = 400)
        print(self.params)
        if save:
            self.save_model()

        return

    def RandomForest(self, save = True):
        space = {
            'bootstrap': [True, False],
            'max_depth': [5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, None],
            'max_features': ['sqrt','log2', None],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'min_samples_split': [2,5,10,12,15,18,21,24],
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
            }

        rf = RandomForestRegressor()
        self.model = RandomizedSearchCV(estimator = rf, param_distributions = space, n_iter = 200, cv = 4,
                                        verbose=1, random_state=random_seed, n_jobs = -1 )
        # Fit the random search model
        self.model = self.model.fit(self.X_test, self.y_test.ravel())
        self.model = self.model.best_estimator_

        self.params = self.model.get_params()

        if save:
            self.save_model()
        
    def save_model(self):
        print("Saving trained model")
        pickle.dump(self.model, open(self.FN_MODEL, 'wb+'))
        if self.PCA_transform:
            pickle.dump(self.pca,   open(self.FN_PCA, 'wb+'))



    def XGBoost(self, save = True):
        space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
                'gamma': hp.uniform ('gamma', 0,15),
                'reg_alpha' : hp.quniform('reg_alpha', 0, 30,1),
                'reg_lambda' : hp.uniform('reg_lambda', 0,4),
                'colsample_bytree' : hp.uniform('colsample_bytree', 0.3,2),
                'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1),
                'n_estimators': hp.uniform('n_estimators', 20, 1500),
                'seed': seed_ho
            }

        trials = Trials()
        self.xgb_params = fmin(fn = self.objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 500,
                            trials = trials)

        self.xgb_params['max_depth'] = int(self.xgb_params['max_depth'])
        self.xgb_params['reg_alpha'] = int(self.xgb_params['reg_alpha'])
        self.xgb_params['min_child_weight'] = int(self.xgb_params['min_child_weight'])
        self.xgb_params['colsample_bytree'] = int(self.xgb_params['colsample_bytree'])
        self.model = xgb.XGBRegressor(
                    max_depth = self.xgb_params['max_depth'], gamma = self.xgb_params['gamma'],
                    reg_alpha = self.xgb_params['reg_alpha'],min_child_weight=self.xgb_params['min_child_weight'],
                    colsample_bytree=self.xgb_params['colsample_bytree'],
                    eval_metric="rmse")
        self.model.fit(self.X_train,self.y_train)
        #print("Real MSE:", (self.y_test**2).mean())
        #self.model = self.model.get_booster()
        self.params = self.model.get_params()

        if save:
            self.save_model()




    def objective(self, space):
        regr=xgb.XGBRegressor(
                        max_depth = int(space['max_depth']), gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']),
                        eval_metric="rmse")

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        evaluation = [(X_train, y_train), (X_test, y_test)]
        
        regr.fit(X_train, y_train, eval_set=evaluation,verbose=False)
        

        pred = regr.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        #print ("MSE:", mse)
        return {'loss': mse, 'status': STATUS_OK }

    def SAGE(self):
        
        PATH_SAGE = os.path.join(self.PATH_PLOTS, "Explainable/")
        if not os.path.exists(PATH_SAGE):
            os.makedirs(PATH_SAGE)

        size = 128
        imputer = sage.MarginalImputer(self.model, self.X_test[:size])
        estimator = sage.PermutationEstimator(imputer, 'mse')
        sage_testues = estimator(self.X_test[:size], self.y_test[:size])

        sage_testues.plot(feature_names=self.xcols)
        plt.tight_layout()

        if self.PCA_transform:
            plt.savefig(os.path.join(PATH_SAGE, f"SAGE_{self.name}_pca_{self.target}.png"), dpi = 400)
        else:
            plt.savefig(os.path.join(PATH_SAGE, f"SAGE_{self.name}_{self.target}.png"), dpi = 400)

    def SHAP(self):

        PATH_SHAP = os.path.join(self.PATH_PLOTS, f"Explainable/SHAP_{self.name}_{self.target}.png")
        if not os.path.exists(PATH_SHAP):
            os.makedirs(PATH_SHAP)

        size = 128
        imputer = sage.MarginalImputer(self.model.get_booster(), self.X_test.values[:size])
        estimator = sage.PermutationEstimator(imputer, 'mse')
        sensitivity = estimator(self.X_test.values[:size])
        # Plot results
        sensitivity.plot(feature_names=self.X_train.columns, title='Model Sensitivity')
        plt.savefig(os.path.join(PATH_SHAP, f'SHAP_{self.name}_{self.target}.png'), dpi = 400)

class XGBoost():
    #write xgboost class definition
    def 



if __name__ == "__main__":
    PATH_FEATURES = './Data/merged_features_all.csv'
    mods = ['XGBoost', 'RandomForest']#, 'NN']
    targs = ['az', 'el', 'total']

    df_mse = pd.DataFrame() # MSE, MODEL, TARGET, FEATURES, PATCH
    idx = 0
    for m in mods:
        for t in targs:
            for col_key in feature_lists.keys():
                for patch_key in patches.keys():
                    row = {}
                    model = Model(df_path = PATH_FEATURES, method = m, target = t, load_model = False, selected_columns_key=col_key, patch_key=patch_key, instrument= 'NFLASH230')
                    model.train(save=True)
                    model.plotSortedPred()

                    row['mse']      = model.evaluate()
                    row['method']   = m
                    row['target']   = t
                    row['features'] = col_key
                    row['patch']    = patch_key
                    print(row)
                    
                    row = pd.DataFrame(_row, index = [0])
                    df_mse = pd.concat([df_mse, row], ignore_index=True)


    df_mse.to_csv('./Results/Results.csv', index = False)

