import pandas as pd
import numpy as np
from tqdm import tqdm
from data_processing import DataProcessing
from functions import get_scan_flags
from IPython import embed
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from move_img import move_img
random_seed = 412069413


class ScanTimes(DataProcessing):
    def __init__(self, path_df):

        df_pointingTable = pd.read_csv(path_df)
        df_pointingTable['obs_date'] = pd.to_datetime(df_pointingTable['obs_date'])
        
        #create new columns
        df_pointingTable['start_observing'] = np.nan
        df_pointingTable['end_observing'] = np.nan
        df_pointingTable['accurateScanTime'] = np.nan
        
        self.df_pointingTable = df_pointingTable




    def get_scan_times(self):
        """
        Get the start and end times for each scan.
        """
        estimateType = 'lower'

        df_durations = pd.DataFrame(data = {'date':self.df_pointingTable.obs_date})
        df_durations['start_observing'] = np.nan
        df_durations['end_observing'] = np.nan
        df_durations['accurateScanTime'] = np.nan
        
        df_scanDuration = pd.read_csv('./Data/scan_durations2022.csv')
        df_scanDuration['date'] = pd.to_datetime(df_scanDuration['date'])


        for rowScan in tqdm(self.df_pointingTable.itertuples()):

            accurateScanTime = True
            scanFlagsCurrentScan = df_scanDuration.loc[(df_scanDuration['date'] == rowScan.obs_date) & (df_scanDuration['rx'] == rowScan.rx)]

            if not scanFlagsCurrentScan.empty:
                start_observing, end_observing = scanFlagsCurrentScan.start.values[0], scanFlagsCurrentScan.end.values[0]
            else:
                start_observing = None
            
            if start_observing is None:
                start_observing = rowScan.obs_date + self.get_scan_diff_mean(rowScan.rx, estimateType=estimateType)
                end_observing = start_observing + self.get_scan_duration_mean(rowScan.rx, estimateType=estimateType)
                accurateScanTime = False

        
            df_durations.loc[rowScan.Index, 'start_observing'] = start_observing
            df_durations.loc[rowScan.Index, 'end_observing'] = end_observing
            df_durations.loc[rowScan.Index, 'accurateScanTime'] = accurateScanTime

        embed()
        df_durations['start_observing'] = pd.to_datetime(df_durations.start_observing)
        df_durations['end_observing'] = pd.to_datetime(df_durations.end_observing)

        
        df_durations['diff'] = (df_durations.start_observing - df_durations.date) / pd.Timedelta('1s')
        df_durations['duration'] = (df_durations.end_observing - df_durations.start_observing) / pd.Timedelta('1s') 

        df_durations.to_csv('./Data/df_scanDuration.csv', index=False)


    





def clean_pointing_table(df):
    """
    Clean the pointing table by removing rows that are not useful for training the model.
    These include very noise scans, test tracking, holo transmitter, etc. 
    """
    ignore_sources = ['Moon', 'HoloTransmitter', 'Tracking-test', 'Tracking-testHOG327', 'Tracking-testHOG327_B']
    
    df = df[df.obs_date >= pd.Timestamp('2022-03-01')]
    df = df[df.obs_date <= pd.Timestamp('2022-09-17 17:30:00')]
    df = df[~df.source.isin(ignore_sources)]
    df = df[~df.rx.isin(['ZEUS2', 'CHAMP690'])]

    return df


def quality_metric_test(df, var):
    """
    Make QM and test it on test set.
    var is "Az" or "El"
    """
    if var.lower() == 'az':
        otherVar = 'El'
    elif var.lower() == 'el':
        otherVar = 'Az'

    dir_good_var = os.listdir(f'./PointingScanPlots/good_{var.lower()}')
    dir_good_both = os.listdir('./PointingScanPlots/good_both')
    dir_bad_both = os.listdir('./PointingScanPlots/bad_both')
    dir_bad_var = os.listdir(f'./PointingScanPlots/good_{otherVar.lower()}')

    good_var  = [filename.split('_')[2] for filename in dir_good_var]
    good_both = [filename.split('_')[2] for filename in dir_good_both]
    bad_both  = [filename.split('_')[2] for filename in dir_bad_both]
    bad_var   = [filename.split('_')[2] for filename in dir_bad_var]

    df['beamsize'] = 7.8 * 800 / df.freq
    df = df[(df[f'Off_e{var.capitalize()}'] < df.beamsize)]

    embed()

    print(df[df.bad_var].loc[df.scan.isin(set(good_var + good_both))])



def transform_offset_and_corrections(df):
    """
    Transform the offsets and corrections what they would look like
    if corrections were made after every scan.
    """

    times = df.obs_date.values
    off_az = df.Off_Az.values
    off_el = df.Off_El.values
    ca = df.ca.values 
    ie = df.ie.values

    timeLastCorrection = [np.nan]
    off_az_new = []
    off_el_new = []
    ca_new = []
    ie_new = []

    off_az_new.append(off_az[0])
    off_el_new.append(off_el[0])
    ca_new.append(ca[0])
    ie_new.append(ie[0])

    for i in range(1, len(off_az)):
        if times[i] == times[i-1]:
            ca_new.append(ca_new[-1])
            ie_new.append(ie_new[-1])
            off_az_new.append(off_az_new[-1])
            off_el_new.append(off_el_new[-1])
            timeLastCorrection.append(times[i-2])


        else:
            ca_new.append(ca_new[i-1] + off_az_new[i-1])
            ie_new.append(ie_new[i-1] - off_el_new[i-1])
            off_az_new.append(off_az[i] + ca[i] - ca_new[i])
            off_el_new.append(off_el[i] - ie[i] + ie_new[i])
            timeLastCorrection.append(times[i-1])


    
    #add the new columns to df
    df['Off_Az_new'] = off_az_new
    df['Off_El_new'] = off_el_new
    df['ca_new'] = ca_new
    df['ie_new'] = ie_new
    df['timeLastCorrection'] = timeLastCorrection
    df.dropna(inplace=True)

    return df


def merge_start_observing_with_pointing_table(df_pointing, df_scans):
    pass


#import logsistic regression from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import PrecisionRecallDisplay

def NN_Classifier(X_train_az, y_train_az, X_train_el, y_train_el):
    # Define hyperparameters to test
    hyperparameters = {
        'hidden_layer_sizes': [(64,128,64), (128, 64, 128), (156,128,156), (200,100,200), ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, 0.1, 1.0],
        #'learning_rate': ['adaptive'],
    }

    # Fit MLPClassifier models with different hyperparameters for az and el
    clf_az = GridSearchCV(MLPClassifier(max_iter=5000), hyperparameters).fit(X_train_az[:, 1:], y_train_az)
    clf_el = GridSearchCV(MLPClassifier(max_iter=5000), hyperparameters).fit(X_train_el[:, 1:], y_train_el)

    return clf_az, clf_el

def logreg_classifier(X_train_az, y_train_az, X_train_el, y_train_el):
    # Define hyperparameters to test
    hyperparameters = {'C': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 150.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']}

    # Fit logistic regression models with different hyperparameters for az and el
    clf_az = GridSearchCV(LogisticRegression(), hyperparameters).fit(X_train_az[:,1:], y_train_az)
    clf_el = GridSearchCV(LogisticRegression(), hyperparameters).fit(X_train_el[:,1:], y_train_el)

    return clf_az, clf_el

def NN_Kfold(X_train_az, y_train_az, X_train_el, y_train_el):
    # Define hyperparameters to test
    hyperparameters = {
        'hidden_layer_sizes': [(64,128,64), (128, 64, 128), (156,128,156), (200,100,200), ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, 0.1, 1.0],
        #'learning_rate': ['adaptive']
        }

    # Define KFold with 5 folds
    kf = KFold(n_splits=4)

    # Fit MLP classifier models with different hyperparameters for az and el
    clf_az = GridSearchCV(MLPClassifier(max_iter=2000), hyperparameters, cv=kf).fit(X_train_az[:,1:], y_train_az)
    clf_el = GridSearchCV(MLPClassifier(max_iter=2000), hyperparameters, cv=kf).fit(X_train_el[:,1:], y_train_el)

    return clf_az, clf_el

def logreg_cleaning(df):
    dir_good_az = os.listdir(f'./PointingScanPlots/good_az')
    dir_good_el = os.listdir(f'./PointingScanPlots/good_el')
    dir_good_both = os.listdir('./PointingScanPlots/good_both')
    dir_bad_both = os.listdir('./PointingScanPlots/bad_both')

    good_both = [filename.split('_')[2] for filename in dir_good_both]
    bad_both  = [filename.split('_')[2] for filename in dir_bad_both]

    good_az   = [filename.split('_')[2] for filename in dir_good_az] + good_both
    good_el   = [filename.split('_')[2] for filename in dir_good_el] + good_both

    bad_az = good_az + bad_both
    bad_el = good_el + bad_both

    scans_az = np.array(good_az + bad_az).astype(int)
    scans_el = np.array(good_el + bad_el).astype(int)
 
    good_az = np.array(good_az).astype(int)
    good_el = np.array(good_el).astype(int)

    df['Amp_rAz'] = df.Amp_Az / df.Amp_eAz
    df['Amp_rEl'] = df.Amp_El / df.Amp_eEl
    df['FWHM_rAz'] = df.FWHM_Az / df.FWHM_eAz
    df['FWHM_rEl'] = df.FWHM_El / df.FWHM_eEl
    df['Off_rAz'] = df.Off_Az / df.Off_eAz
    df['Off_rEl'] = df.Off_El / df.Off_eEl

    features_az = ['scan', 'Off_Az', 'Off_eAz', 'FWHM_Az', 'FWHM_eAz', 'Amp_Az', 'Amp_eAz', 'beamsize'] #'Amp_rAz', 'FWHM_rAz', 'Off_rAz']
    features_el = ['scan', 'Off_El', 'Off_eEl', 'FWHM_El', 'FWHM_eEl', 'Amp_El', 'Amp_eEl', 'beamsize'] #'Amp_rEl', 'FWHM_rEl', 'Off_rEl']
    
    data_az = df.loc[df.scan.isin(scans_az)]
    data_el = df.loc[df.scan.isin(scans_el)]
    
    dummies_rx_az = pd.get_dummies(data_az['rx'])
    dummies_rx_el = pd.get_dummies(data_el['rx'])



    data_az = data_az[features_az]
    data_el = data_el[features_el]

    # data_az = pd.concat([data_az, dummies_rx_az], axis=1)
    # data_el = pd.concat([data_el, dummies_rx_el], axis=1)
    
    data_az['good'] = np.where(data_az.scan.isin(good_az), 1, 0)
    data_el['good'] = np.where(data_el.scan.isin(good_el), 1, 0)

    

    # split into train and test set
    X_train_az, X_test_az, y_train_az, y_test_az = train_test_split(data_az.iloc[:, :-1], data_az['good'], test_size=0.2, random_state=0)
    X_train_el, X_test_el, y_train_el, y_test_el = train_test_split(data_el.iloc[:, :-1], data_el['good'], test_size=0.2, random_state=0)

    # combine the training data and split into good and bad scans
    data_az_train = pd.concat([X_train_az, y_train_az], axis=1)
    data_el_train = pd.concat([X_train_el, y_train_el], axis=1)
    good_scans_az_train = data_az_train[data_az_train.good == True]
    bad_scans_az_train = data_az_train[data_az_train.good == False]
    good_scans_el_train = data_el_train[data_el_train.good == True]
    bad_scans_el_train = data_el_train[data_el_train.good == False]

    # upsample the bad scans in the training data to match the number of good scans
    bad_scans_az_upsampled = resample(bad_scans_az_train, replace=True, n_samples=len(good_scans_az_train))
    bad_scans_el_upsampled = resample(bad_scans_el_train, replace=True, n_samples=len(good_scans_el_train))

    # combine the upsampled bad scans with the good scans to form the new training data
    data_az_train_upsampled = pd.concat([good_scans_az_train, bad_scans_az_upsampled])
    data_el_train_upsampled = pd.concat([good_scans_el_train, bad_scans_el_upsampled])

    # extract the features and labels from the upsampled training data
    X_train_az_upsampled = data_az_train_upsampled.iloc[:, :-1].values
    y_train_az_upsampled = data_az_train_upsampled['good'].values
    X_train_el_upsampled = data_el_train_upsampled.iloc[:, :-1].values
    y_train_el_upsampled = data_el_train_upsampled['good'].values

    X_train_az = X_train_az_upsampled
    y_train_az = y_train_az_upsampled
    X_train_el = X_train_el_upsampled
    y_train_el = y_train_el_upsampled
    
    # extract the features and labels from the test data
    X_test_az = X_test_az.values
    y_test_az = y_test_az.values
    X_test_el = X_test_el.values
    y_test_el = y_test_el.values


    scaler_az = StandardScaler()
    scaler_el = StandardScaler()
    # Fit the transformers on the training data and transform both training and test data
    X_train_az[:,1:len(features_az)] = scaler_az.fit_transform(X_train_az[:,1:len(features_az)])
    X_test_az[:,1:len(features_az)] = scaler_az.transform(X_test_az[:,1:len(features_az)])

    X_train_el[:,1:len(features_az)] = scaler_el.fit_transform(X_train_el[:,1:len(features_az)])
    X_test_el[:,1:len(features_az)] = scaler_el.transform(X_test_el[:,1:len(features_az)])


    clf_az, clf_el = NN_Kfold(X_train_az, y_train_az, X_train_el, y_train_el)
    # clf_az, clf_el = NN_Classifier(X_train_az, y_train_az, X_train_el, y_train_el)


    # Print the best hyperparameters for each model
    print('Best hyperparameters for az: ', clf_az.best_params_)
    print('Best hyperparameters for el: ', clf_el.best_params_)

    # Print train score, test score, and plot a subplot with confusion matrix for az and el
    print('Train score az: ', clf_az.score(X_train_az[:,1:], y_train_az))
    print('Train score el: ', clf_el.score(X_train_el[:,1:], y_train_el))
    print('Test score az: ', clf_az.score(X_test_az[:,1:], y_test_az))
    print('Test score el: ', clf_el.score(X_test_el[:,1:], y_test_el))

    # Make confusion matrix with sklearn and plot using seaborn
    y_pred_az = clf_az.predict(X_test_az[:,1:])
    y_pred_el = clf_el.predict(X_test_el[:,1:])
    cm_az = confusion_matrix(y_test_az, y_pred_az)
    cm_el = confusion_matrix(y_test_el, y_pred_el)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    sns.heatmap(cm_az, annot = True, fmt = 'd', ax = ax1)
    sns.heatmap(cm_el, annot = True, fmt = 'd', ax = ax2)
    ax1.set_title('Confusion matrix az')
    ax2.set_title('Confusion matrix el')
    plt.savefig('./ClassifyingScans/ConfusionMatrix.png')

    # Print the first column of X_test_az and X_test_el for the the incorrect classifications
    print('Incorrect az scans: ', X_test_az[y_pred_az != y_test_az, 0], y_test_az[y_pred_az != y_test_az])
    print('Incorrect el scans: ', X_test_el[y_pred_el != y_test_el, 0], y_test_el[y_pred_el != y_test_el])




    # Create PrecisionRecallDisplay object for classifier clf_az
    display_az = PrecisionRecallDisplay.from_estimator(
        clf_az, X_test_az[:,1:], y_test_az, name="clf_az"
    )

    # Create PrecisionRecallDisplay object for classifier clf_el
    display_el = PrecisionRecallDisplay.from_estimator(
        clf_el, X_test_el[:,1:], y_test_el, name="clf_el"
    )

    # Plot both P-R curves together
    fig, ax = plt.subplots()
    display_az.plot(ax=ax)
    display_el.plot(ax=ax)
    ax.set_title("2-class Precision-Recall curve")
    ax.legend()
    plt.savefig("./ClassifyingScans/precision_recall.png")

    return clf_az, clf_el


def classify_new_scans(PATH_MODEL, df, threshold = 0.5, move_imgs = False):
    clf = pickle.load(open(PATH_MODEL, 'rb'))

    var = PATH_MODEL.split('/')[-1].split('.')[0].split('_')[1] # az or el

    if var == 'az':
        otherVar = 'el'
    elif var == 'el':
        otherVar = 'az'
    else:
        raise ValueError('var must be az or el')    
    #Find the good and bad scans
    dir_good = os.listdir(f'./PointingScanPlots/good_{var}')
    dir_bad = os.listdir(f'./PointingScanPlots/good_{otherVar}')
    dir_good_both = os.listdir('./PointingScanPlots/good_both')
    dir_bad_both = os.listdir('./PointingScanPlots/bad_both')

    good_both = [filename.split('_')[2] for filename in dir_good_both]
    bad_both  = [filename.split('_')[2] for filename in dir_bad_both]

    good  = [filename.split('_')[2] for filename in dir_good] + good_both
    bad   = [filename.split('_')[2] for filename in dir_bad] + bad_both

    scans = np.array(good + bad).astype(int)
    good = np.array(good).astype(int)

    df = df[~df['scan'].isin(scans)]

    #Sample 100 random scans
    #df = df.sample(100, random_state = random_seed)

    features = ['scan','beamsize','rx',
        f'Off_{var.capitalize()}',
        f'Off_e{var.capitalize()}',
        f'FWHM_{var.capitalize()}',
        f'FWHM_e{var.capitalize()}',
        f'Amp_{var.capitalize()}',
        f'Amp_e{var.capitalize()}'
        ]
    #Get the features
    df = df[features]

    n_features = len(features)

    if 'rx' in features:
        le = LabelEncoder()
        df['rx'] = le.fit_transform(df['rx'])

    X = df.iloc[:,1:].values

    #Predict the classes
    y_probs = clf.predict_proba(X)
    y_pred = np.where(y_probs[:,1] > threshold, 1, 0)

    path_good = f'./ClassifyingScans/Classified/good_{var}'
    path_bad = f'./ClassifyingScans/Classified/bad_{var}'
    for _path in [path_good, path_bad]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    if move_imgs:
        for i, scan in enumerate(df['scan']):
            if y_pred[i] == 1:
                _path = path_good
            elif y_pred[i] == 0:
                _path = path_bad

            move_img(scan, PATH_MOVE = _path)

    good_scans = df[y_pred == 1]['scan'].values
    
    return good_scans


def classify_new_scans_v2(PATH_MODEL, df, threshold=0.5, move_imgs = False):
    clf = pickle.load(open(PATH_MODEL, 'rb'))
    var = PATH_MODEL.split('/')[-1].split('.')[0].split('_')[1] # az or el


    #Find the good and bad scans
    dir_good = os.listdir(f'./PointingScanPlots_v2/Good')
    dir_bad = os.listdir(f'./PointingScanPlots_v2/Bad')

    good  = [filename.split('_')[2] for filename in dir_good]
    bad   = [filename.split('_')[2] for filename in dir_bad]

    scans = np.array(good + bad).astype(int)
    good = np.array(good).astype(int)

    df = df[~df['scan'].isin(scans)]

    #Sample 100 random scans
    #df = df.sample(100, random_state = random_seed)
    
    features = ['scan','beamsize',
            'Off_Az', 'Off_eAz', 'FWHM_Az', 'FWHM_eAz', 'Amp_Az', 'Amp_eAz',
            'Off_El', 'Off_eEl', 'FWHM_El', 'FWHM_eEl', 'Amp_El', 'Amp_eEl'
            ]
    
    #Get the features
    df = df[features]

    n_features = len(features)

    if 'rx' in features:
        le = LabelEncoder()
        df['rx'] = le.fit_transform(df['rx'])

    X = df.iloc[:,1:].values

    #Predict the classes
    y_probs = clf.predict_proba(X)
    y_pred = np.where(y_probs[:,1] > threshold, 1, 0)

    path_good = f'./ClassifyingScans_v2/Classified/Good'
    path_bad = f'./ClassifyingScans_v2/Classified/Bad'
    for _path in [path_good, path_bad]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    if move_imgs:
        for i, scan in enumerate(df['scan']):
            if y_pred[i] == 1:
                _path = path_good
            elif y_pred[i] == 0:
                _path = path_bad

            move_img(scan, PATH_MOVE = _path)

    good_scans = df[y_pred == 1]['scan'].values
    bad_scans = df[y_pred == 0]['scan'].values

    return good_scans, bad_scans

def get_good_scans_from_training_data():
    dir_good_az = os.listdir(f'./PointingScanPlots/good_az')
    dir_good_el = os.listdir(f'./PointingScanPlots/good_el')
    dir_good_both = os.listdir('./PointingScanPlots/good_both')
    dir_bad_both = os.listdir('./PointingScanPlots/bad_both')

    good_both = [filename.split('_')[2] for filename in dir_good_both]
    good_az   = [filename.split('_')[2] for filename in dir_good_az] + good_both
    good_el   = [filename.split('_')[2] for filename in dir_good_el] + good_both

    good_scans_az = np.array(good_az).astype(int)
    good_scans_el = np.array(good_el).astype(int)
    return good_az, good_el


def get_good_scans_from_training_data_v2():
    dir_good = os.listdir(f'./PointingScanPlots_v2/Good')
    dir_bad = os.listdir(f'./PointingScanPlots_v2/Bad')

    good = [filename.split('_')[2] for filename in dir_good]
    bad  = [filename.split('_')[2] for filename in dir_bad]

    return good, bad

def dataset_for_finetuning(path_scans,path_features,name=''):
    df = pd.read_csv(path_scans)
    print(len(df))
    df = df[df.rx == 'NFLASH230']
    print(len(df))
    df2 = pd.read_csv(path_features)

    df['obs_date'] = pd.to_datetime(df['obs_date'])
    df = df[['obs_date', 'Off_Az', 'Off_El']]
    df2['date'] = pd.to_datetime(df2['date'])

    df = df.rename(columns={'obs_date':'date'})

    df = df.drop_duplicates(subset='date')
    df2 = df2.drop_duplicates(subset='date')
    
    #merge df and df2 on date.
    df3 = pd.merge(df, df2, on='date', how='inner')

    #print number of nans in each column
    df3 = df3.dropna(thresh=len(df3)-30, axis = 1).dropna()

    df3 = df3[(df3.COMMANDAZ_MEDIAN != 0.0) | (df3.COMMANDEL_MEDIAN != 0.0)]
    df3['azdiff'] = (df3.COMMANDAZ_MEDIAN - df3.ACTUALAZ_MEDIAN).abs()
    df3['eldiff'] = (df3.COMMANDEL_MEDIAN - df3.ACTUALEL_MEDIAN).abs()

    threshold_az = df3.azdiff.quantile(0.97)
    threshold_el = df3.eldiff.quantile(0.97)

    df3 = df3[(df3.azdiff < threshold_az) & (df3.eldiff < threshold_el)]

    df3['REALAZ'] = df3['ACTUALAZ_MEDIAN'] - np.deg2rad(df3['Off_Az'] / 3600) * np.cos(df3['ACTUALEL_MEDIAN'])
    df3['REALEL'] = df3['ACTUALEL_MEDIAN'] - np.deg2rad(df3['Off_El'] / 3600)
    
    df3.to_csv(f'./Data/{name}.csv', index=False)



def make_all_datasets_for_thesis():
    path_tmp = './Data/tmp2022.csv'

    df = pd.read_csv(path_tmp)
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    df = df.sort_values(by='obs_date')
    df['beamsize'] = 7.8 * 800 / df['freq']
    df = df.drop_duplicates()

    df_cleaned = clean_pointing_table(df)
    df_cleaned.to_csv('./Datasets/tmp2022_clean.csv', index=False)
    
    df_cleaned_nflash230 = df_cleaned[df_cleaned.rx == 'NFLASH230']
    df_cleaned_nflash230.to_csv('./Datasets/tmp2022_clean_nflash230.csv', index=False)

    #Remove scans with xgbclassifier
    path_clf = './ClassifyingScans_v2/Models/XGB_both_rx.pkl'
    good_scans, bad_scans = classify_new_scans_v2(path_clf, df_cleaned, threshold = 0.8, move_imgs = False)

    df_cleaned_clf = df_cleaned[df_cleaned.scan.isin(good_scans)]
    df_cleaned_clf.to_csv('./Datasets/tmp2022_clean_clf.csv', index=False)

    df_cleaned_clf_nflash230 = df_cleaned_clf[df_cleaned_clf.rx == 'NFLASH230']
    df_cleaned_clf_nflash230.to_csv('./Datasets/tmp2022_clean_clf_nflash230.csv', index=False)

    #Transform offsets and corrections
    df_cleaned_transformed = transform_offset_and_corrections(df_cleaned)
    df_cleaned_transformed.to_csv('./Datasets/tmp2022_clean_transformed.csv', index=False)

    df_cleaned_nflash230_transformed = transform_offset_and_corrections(df_cleaned_nflash230)
    df_cleaned_nflash230_transformed.to_csv('./Datasets/tmp2022_clean_nflash230_transformed.csv', index=False)

    df_cleaned_clf_transformed = transform_offset_and_corrections(df_cleaned_clf)
    df_cleaned_clf_transformed.to_csv('./Datasets/tmp2022_clean_clf_transformed.csv', index=False)

    df_cleaned_clf_nflash230_transformed = transform_offset_and_corrections(df_cleaned_clf_nflash230)
    df_cleaned_clf_nflash230_transformed.to_csv('./Datasets/tmp2022_clean_clf_nflash230_transformed.csv', index=False)

def add_offsets_and_corrections_to_datasets():
    dir_path = "./Datasets/"

    # Loop through each CSV file in the directory
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".csv"):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(dir_path, file_name))
            df.rename(columns={"obs_date": "date"}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
    
            # Set the subfolder path based on the file name
            subfolder_path = os.path.join(dir_path, file_name[:-4])
            
            features_df = pd.read_csv(os.path.join(subfolder_path, "features.csv"))
            features_df['date'] = pd.to_datetime(features_df['date'])

            if file_name.endswith("transformed.csv"):
                merged_df = pd.merge(features_df, df[['date', 'rx', 'ca_new', 'ie_new', 'Off_Az_new', 'Off_El_new']], how='right', left_on='date', right_on='date')
                merged_df.rename(columns={"ca_new": "ca", "ie_new": "ie", "Off_Az_new": "Off_Az", "Off_El_new": "Off_El"}, inplace=True)
                merged_df["hours_since_corr"] = (df["date"] - df["date"].shift()).astype('timedelta64[h]')

            else:
                merged_df = pd.merge(features_df, df[['date', 'rx', 'ca', 'ie', 'Off_Az', 'Off_El']], how='right', left_on='date', right_on='date')

            merged_df["month_continuous"] = merged_df.date.dt.month + (merged_df.date.dt.day - 1) / merged_df.date.dt.days_in_month
            merged_df['month'] = merged_df.date.dt.month
            merged_df['time_of_day'] = merged_df.date.dt.hour / 24 + merged_df.date.dt.minute / 1440 + merged_df.date.dt.second / 86400
            merged_df.to_csv(os.path.join(subfolder_path, "features_offsets.csv"), index=False)


if __name__ == '__main__':

    # make_all_datasets_for_thesis()
    # make_all_datasets_for_thesis()
    # add_offsets_and_corrections_to_datasets()

    # df = pd.read_csv('./Data/tmp2022_clean.csv')
    # df['obs_date'] = pd.to_datetime(df['obs_date'])
    # df = df.sort_values(by='obs_date')
    # df = transform_offset_and_corrections(df)
    # df.to_csv('./Data/tmp2022_clean.csv', index=False)
    # path_scans_clean = './Data/tmp2022_clean_v2.csv'
    # path_scans_all = './Data/tmp2022.csv'

    # path_features = './Data/processed_v3/all_features_safe_v2.csv'
    # dataset_for_finetuning(path_scans_all, path_features, name = 'scans_nflash230_unscaled_all')

    path_clf = './ClassifyingScans_v2/Models/XGB_both_rx.pkl'

    #Load xgb model and print parameters
    xgb_clf = pickle.load(open(path_clf, "rb"))
    print(xgb_clf.get_params())
    

    """
    df.to_csv('./Data/scans_nflash230.csv', index=False)
    df = pd.read_csv('tmp2022_clean_v2.csv')
    df2 = pd.read_csv()
    embed()
    path_tmp =  './Data/tmp2022.csv'
    df = pd.read_csv(path_tmp)
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    df = df.sort_values(by='obs_date')
    df['beamsize'] = 7.8 * 800 / df['freq']
    df = clean_pointing_table(df)
    
    # quality_metric_test(df, 'El')
    # print(df)
    good_scans_az_training, good_scans_el_training = get_good_scans_from_training_data()

    good_scans_az = classify_new_scans('./ClassifyingScans/Models/XGB_az.pkl', df, threshold = 0.85, move_imgs = False)
    good_scans_el = classify_new_scans('./ClassifyingScans/Models/XGB_el.pkl', df, threshold = 0.92, move_imgs = False)

    

    df['good_scan_az'] = df['scan'].isin(good_scans_az) | df['scan'].isin(good_scans_az_training)
    df['good_scan_el'] = df['scan'].isin(good_scans_el) | df['scan'].isin(good_scans_el_training)
    df['good_scan'] = df['good_scan_az'] & df['good_scan_el']

    embed()

    df.to_csv('./Data/tmp2022_clean.csv', index=False)


    
    embed()
    """
    # scanTimesObject = ScanTimes(path_tmp)
    # scanTimesObject.get_scan_times()


"""    df = pd.read_csv('./Data/tmp2022.csv')
    df['obs_date'] = pd.to_datetime(df['obs_date'])
    df = clean_pointing_table(df)
    df = transform_offset_and_corrections(df)
    df.to_csv('./Data/tmp2022_clean.csv', index=False)"""