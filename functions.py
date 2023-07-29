from xml.sax.handler import feature_namespaces
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import os
import sys
import ephem
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import xgboost as xgb
import sage
from hyperopt import hp
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pickle

random_seed = 12

plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
sns.set(font_scale = 1.5)

#paths 
path_data = "./Data/"
path_correlations = "./Correlations/"
path_models = "./Plots/"
path_features = "./Features/"
path_pointing = os.path.join(path_data,"./PointingTable.csv")
path_meteoscope = os.path.join(path_data, "./meteoscope.csv")



#Functions
def get_sun_position(date_col):
    apex = ephem.Observer()
    apex.lat  = '-23:00:20.8'
    apex.long = '-67:45:33.0'
    apex.elev =   5105

    date_str = date_col.dt.strftime('%Y/%m/%d %H:%M')

    sun = ephem.Sun()

    Az_sun = np.zeros(len(date_str))
    El_sun = np.zeros(len(date_str))

    for i in range(len(date_str)):
        apex.date = date_str[i]
        sun.compute(apex)
        Az_sun[i] = np.degrees(sun.az)
        El_sun[i] = np.degrees(sun.alt)

    return Az_sun, El_sun

def make_df(df):

	#make new dataframes for each of the unique parameter_id in ms dataframe
	df["date_measured"] = pd.to_datetime(df["date_measured"])
	dewpoint = df[df.parameter_id == -3]
	air_pressure = df[df.parameter_id == -2]
	temperature = df[df.parameter_id == 0]
	relative_humidity = df[df.parameter_id == 1]
	wind_speed = df[df.parameter_id == 5]
	wind_direction = df[df.parameter_id == 6]
	precipitable_water_vapour = df[df.parameter_id == 12]

	#remove parameter_id column from the 7 dataframes above
	dewpoint = dewpoint.drop(columns=['parameter_id'])
	air_pressure = air_pressure.drop(columns=['parameter_id'])
	temperature = temperature.drop(columns=['parameter_id'])
	relative_humidity = relative_humidity.drop(columns=['parameter_id'])
	wind_speed = wind_speed.drop(columns=['parameter_id'])
	wind_direction = wind_direction.drop(columns=['parameter_id'])
	precipitable_water_vapour = precipitable_water_vapour.drop(columns=['parameter_id'])

	dewpoint = dewpoint.rename(columns = {"value_median": "median_dewpoint", "value_mean": "mean_dewpoint", "value_first": "first_dewpoint"})
	air_pressure = air_pressure.rename(columns = {"value_median": "median_air_pressure", "value_mean": "mean_air_pressure", "value_first": "first_air_pressure"})
	temperature = temperature.rename(columns = {"value_median": "median_temperature", "value_mean": "mean_temperature", "value_first": "first_temperature"})
	relative_humidity = relative_humidity.rename(columns = {"value_median": "median_relative_humidity", "value_mean": "mean_relative_humidity", "value_first": "first_relative_humidity"})
	wind_speed = wind_speed.rename(columns = {"value_median": "median_wind_speed", "value_mean": "mean_wind_speed", "value_first": "first_wind_speed"})
	wind_direction = wind_direction.rename(columns = {"value_median": "median_wind_direction", "value_mean": "mean_wind_direction", "value_first": "first_wind_direction"})
	precipitable_water_vapour = precipitable_water_vapour.rename(columns = {"value_median": "median_pwv", "value_mean": "mean_pwv", "value_first": "first_pwv"})

	#join the 7 dataframes above into 1 dataframe, join them on the column "date_measured". Keep only one of value_src and location_id
	df_joined = pd.merge(dewpoint, air_pressure, on=['date_measured', 'value_src', 'location_id'], how='outer')
	df_joined = pd.merge(df_joined, temperature, on=['date_measured', 'value_src', 'location_id'], how='outer')
	df_joined = pd.merge(df_joined, relative_humidity, on=['date_measured', 'value_src', 'location_id'], how='outer')
	df_joined = pd.merge(df_joined, wind_speed, on=['date_measured', 'value_src', 'location_id'], how='outer')
	df_joined = pd.merge(df_joined, wind_direction, on=['date_measured', 'value_src', 'location_id'], how='outer')
	df_joined = pd.merge(df_joined, precipitable_water_vapour, on=['date_measured', 'value_src', 'location_id'], how='outer')
	df_joined.drop_duplicates(inplace=True)
	df_joined["date_measured"] = pd.to_datetime(df_joined["date_measured"])

	return df_joined

def find_min_idx(df_meteoscope, df_pointing):
    time_ps = np.array(df_pointing["obs_date"])
    time_ms = np.array(df_meteoscope["date_measured"])

    diff = time_ps[:, np.newaxis] - time_ms[np.newaxis]
    diff = np.where(diff >= np.timedelta64(0, 's'), diff, 1e8)
    diff_min = np.amin(diff, axis=1)
    #print(diff_min, type(diff_min[0]))
    #print(np.where(diff_min > np.timedelta64(1800000, 'ms')))
    #Contains index for data in meteoscope data that is the closest in time to pointing scan.
    idx_min = np.array([np.where(diff[i] == diff_min[i])[0][0] for i in range(len(diff))])
    return idx_min

def split_data(X, yAz, yEl, test_size=0.2, random_seed = 12):
    X_train, X_test, yAz_train, yAz_test = train_test_split(X.values, yAz.values, test_size=test_size, random_state=random_seed)
    X_train, X_test, yEl_train, yEl_test = train_test_split(X.values, yEl.values, test_size=test_size, random_state=random_seed)

    return X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test

def scale_data(Scaler, X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test):
    X_trainShape = X_train.shape
    X_testShape = X_test.shape

    Scaler.fit(X_train.reshape(-1,1))
    X_train = Scaler.transform(X_train.reshape(-1,1)).reshape(X_trainShape)
    X_test = Scaler.transform(X_test.reshape(-1,1)).reshape(X_testShape)

    yAz_train = Scaler.transform(yAz_train.reshape(-1,1)).reshape(X_trainShape[0])    
    yEl_train = Scaler.transform(yEl_train.reshape(-1,1)).reshape(X_trainShape[0])

    yAz_test = Scaler.transform(yAz_test.reshape(-1,1)).reshape(X_testShape[0])
    yEl_test = Scaler.transform(yEl_test.reshape(-1,1)).reshape(X_testShape[0]) 
    
    return Scaler, X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test


def makedfs(path_meteoscope, path_pointing):
    df_meteoscope = pd.read_csv(path_meteoscope)
    df_meteoscope_joined = make_df(df_meteoscope)
    df_meteoscope_joined["date_measured"] = pd.to_datetime(df_meteoscope_joined["date_measured"])


    df_pointing = pd.read_csv(path_pointing)
    df_pointing = df_pointing[:-3]
    df_pointing["obs_date"] = pd.to_datetime(df_pointing["obs_date"])


    min_idx = find_min_idx(df_meteoscope_joined, df_pointing)

    #Joined meteoscope df with only data closest to pointing observation
    df_meteoscope_obs = df_meteoscope_joined.iloc[min_idx]

    return df_meteoscope, df_meteoscope_joined, df_meteoscope_obs, df_pointing, min_idx

def linearInterpolation(df_meteoscope_joined, df_meteoscope_obs, df_pointing, min_idx):
    #Change between observations. Used to linearly interpolate
    diff_meteoscope_joined = df_meteoscope_joined.drop(labels=["location_id", "value_src"], axis = 1).diff()
    diff_meteoscope_obs = diff_meteoscope_joined.iloc[min_idx]

    #Timedifference between observation and weather data
    dt = np.array(df_pointing["obs_date"]).astype("datetime64[s]")-np.array(df_meteoscope_obs["date_measured"]).astype("datetime64[s]")
    dt = np.repeat(dt[:,np.newaxis], len(diff_meteoscope_obs.columns), axis=1)

    #Linearly interpolate meteoscope data. This df now contains the full df but with intetrpolated values based on observation time.
    df_meteoscope_linint = df_meteoscope_obs + diff_meteoscope_obs*(dt/np.timedelta64(3600,'s'))
    df_meteoscope_linint.reset_index(drop=True, inplace=True)

    return df_meteoscope_linint

def transformData(df_meteoscope_linint, df_pointing):

    #Set all missing values to the mean median pwv. Later improvement could be checking if the other pwv columns have values where median is missing.
    mean_median_pwv = df_meteoscope_linint["median_pwv"][(df_meteoscope_linint["median_pwv"] > 0) & (df_meteoscope_linint["median_pwv"] < 50)].mean()
    df_meteoscope_linint["median_pwv"] = np.where(df_meteoscope_linint["median_pwv"] < 0, mean_median_pwv, df_meteoscope_linint["median_pwv"])

    #time of day array
    hour = df_meteoscope_linint["date_measured"].dt.hour# + df_meteoscope_linint["date_measured"].dt.minute/60 + df_meteoscope_linint["date_measured"].dt.second/3600

    #Sun above horizon categorical variable
    Az_sun, El_sun = get_sun_position(df_meteoscope_linint["date_measured"])
    Az_sun = Az_sun - 180
    sunAboveHorizon = np.where(El_sun > 0, 1, 0)

    #Smallest angle between sun and pointing direction
    smallestAngleSun = calcSmallestAngle(df_pointing["Az"], df_pointing["El"], Az_sun, El_sun)

    #Wind direction compared to pointing
    df_meteoscope_linint["mean_wind_direction"] = df_meteoscope_linint["mean_wind_direction"]-180
    wind_direction_diff = df_meteoscope_linint["mean_wind_direction"] - df_pointing["Az"]

    #Sun position relative to pointing
    sunAz_diff = Az_sun - df_pointing["Az"]
    sunEl_diff = El_sun - df_pointing["El"]

    X_ = np.array([
        df_pointing["Off_Az"],
        df_pointing["Off_El"],    
        hour,
        df_pointing["Az"],
        df_pointing["El"],
        sunAboveHorizon,
        Az_sun,
        El_sun,
        sunAz_diff,
        sunEl_diff,
        smallestAngleSun,
        df_meteoscope_linint["mean_dewpoint"],
        df_meteoscope_linint["mean_air_pressure"],
        df_meteoscope_linint["mean_temperature"],
        df_meteoscope_linint["mean_relative_humidity"],
        df_meteoscope_linint["mean_wind_speed"],
        df_meteoscope_linint["mean_wind_direction"],
        wind_direction_diff,
        df_meteoscope_linint["median_pwv"]
        ]).T
    
    return X_

def createMerged(writeFile = False):
	df2 = pd.read_csv("./Data/DF2_old.csv")
	ts = pd.read_csv("./Data/pointingScans2022.txt", header=None, names=["time"])
	dfp = pd.read_csv("./Data/PointingTable.csv")
	dfp["obs_date"] = pd.to_datetime(dfp["obs_date"])
	time = pd.to_datetime(ts["time"])

	df2["date"] = pd.to_datetime(df2["date"])
	colTypes = ["f", "l", "c", "v"]
	colNames = [_col + "_" + _type for _col in df2.columns[1:] for _type in colTypes]
	colRads = ["ACTUALAZ", "ACTUALEL"]
	for _col in colRads:
		df2[_col] = np.rad2deg(df2[_col])

	df2["WINDDIRECTION"] -= 180

	compressed = {}
	for t in time:
		df_scan = df2[(t - pd.Timedelta(1,unit="m") <= df2["date"]) & (df2["date"] <= t + pd.Timedelta(1,unit="m"))]
		df_scan.drop_duplicates(inplace = True)
		cols = []
		for col in df_scan.columns[1:]:
			first = df_scan[col][df_scan[col].notna()].iloc[0]
			last = df_scan[col][df_scan[col].notna()].iloc[-1]
			change = last - first
			var = df_scan[col][df_scan[col].notna()].var()
			
			cols.extend([first, last, change, var])
		compressed[t] = cols
	
	df_compressed = pd.DataFrame.from_dict(compressed, orient = "index", columns= colNames)
	df_compressed.reset_index(inplace=True)
	df_compressed = df_compressed.rename(columns = {'index':'obs_date'})
	df_compressed.drop_duplicates(inplace = True)

	df_merged = pd.merge(dfp[["obs_date", "Az", "El", "freq", "tint", "sigfi", "fres", "Off_eAz", "Off_eEl","Off_Az", "Off_El"]], df_compressed, on="obs_date", how="inner")

	# Adding hour of day as to df
	df_merged["obs_date"] = pd.to_datetime(df_merged["obs_date"])
	df_merged["hour"] = df_merged["obs_date"].dt.hour
	print(df_merged.columns)
	#Total offset
	df_merged.insert(loc=9, column='total_offset', value=np.sqrt(df_merged["Off_Az"]**2 + df_merged["Off_El"]**2))
	# Transform columns to (-180,180) degree scale
	df_merged.loc[df_merged["ACTUALAZ_f"] > 180, "ACTUALAZ_f"] -= 360
	df_merged.loc[df_merged["ACTUALAZ_f"] < -180, "ACTUALAZ_f"] += 360
	df_merged.loc[df_merged["ACTUALAZ_l"] > 180, "ACTUALAZ_l"] -= 360
	df_merged.loc[df_merged["ACTUALAZ_l"] < -180, "ACTUALAZ_l"] += 360

	# Sun position and sun above horizon 
	Az_sun, El_sun = get_sun_position(df_merged["obs_date"])
	Az_sun = Az_sun - 180
	sunAboveHorizon = np.where(El_sun > 0, 1, 0)

	# Smallest angle between sun and pointing direction
	smallestAngleSunFirst = calcSmallestAngle(df_merged["ACTUALAZ_f"], df_merged["ACTUALEL_f"], Az_sun, El_sun)
	smallestAngleSunLast = calcSmallestAngle(df_merged["ACTUALAZ_l"], df_merged["ACTUALEL_l"], Az_sun, El_sun)
		
	# Wind direction compared to pointing direction
	df_merged["WINDDIRECTION_f_diff"] = df_merged["WINDDIRECTION_f"]-df_merged["ACTUALAZ_f"]
	df_merged["WINDDIRECTION_l_diff"] = df_merged["WINDDIRECTION_l"]-df_merged["ACTUALAZ_l"]

	#Scale columns to (-180,180) degree scale
	df_merged.loc[df_merged["WINDDIRECTION_f_diff"] > 180, "WINDDIRECTION_f_diff"] -= 360
	df_merged.loc[df_merged["WINDDIRECTION_f_diff"] < -180, "WINDDIRECTION_f_diff"] += 360
	df_merged.loc[df_merged["WINDDIRECTION_l_diff"] > 180, "WINDDIRECTION_l_diff"] -= 360
	df_merged.loc[df_merged["WINDDIRECTION_l_diff"] < -180, "WINDDIRECTION_l_diff"] += 360


	# Sun position relative to pointing, and smallest angle to sun
	df_merged["Az_sun"] = Az_sun
	df_merged["El_sun"] = El_sun
	df_merged["sunAboveHorizon"] = sunAboveHorizon
	df_merged["smallestAngleSunFirst"] = smallestAngleSunFirst
	df_merged["smallestAngleSunLast"] = smallestAngleSunLast
	df_merged["sunAz_diff_f"] = Az_sun - df_merged["ACTUALAZ_f"]
	df_merged["sunAz_diff_l"] = Az_sun - df_merged["ACTUALAZ_l"]
	df_merged["sunEl_diff_f"] = El_sun - df_merged["ACTUALEL_f"]
	df_merged["sunEl_diff_l"] = El_sun - df_merged["ACTUALEL_l"]

	#Scale columns to (-180,180) degree scale
	df_merged.loc[df_merged["sunAz_diff_f"] > 180, "sunAz_diff_f"] -= 360
	df_merged.loc[df_merged["sunAz_diff_f"] < -180, "sunAz_diff_f"] += 360
	df_merged.loc[df_merged["sunAz_diff_l"] > 180, "sunAz_diff_l"] -= 360
	df_merged.loc[df_merged["sunAz_diff_l"] < -180, "sunAz_diff_l"] += 360

	cols = list(df_merged.columns)
	idxs = cols[9:12]+cols[7:9]+cols[:7] + cols[12:]
	df_merged = df_merged.loc[:,idxs]
	
	if writeFile:
		df_merged.to_csv("./Data/DF2_merged.csv", index=False)



def transformData2(df_merged):
	# Adding hour of day as to df
	df_merged["obs_date"] = pd.to_datetime(df_merged["obs_date"])
	df_merged["hour"] = df_merged["obs_date"].dt.hour

	#Total offset
	df_merged.insert(loc=9, column='total_offset', value=np.sqrt(df_merged["Off_Az"]**2 + df_merged["Off_El"]**2))
	# Transform columns to (-180,180) degree scale
	df_merged.loc[df_merged["ACTUALAZ_f"] > 180, "ACTUALAZ_f"] -= 360
	df_merged.loc[df_merged["ACTUALAZ_f"] < -180, "ACTUALAZ_f"] += 360
	df_merged.loc[df_merged["ACTUALAZ_l"] > 180, "ACTUALAZ_l"] -= 360
	df_merged.loc[df_merged["ACTUALAZ_l"] < -180, "ACTUALAZ_l"] += 360

	# Sun position and sun above horizon 
	Az_sun, El_sun = get_sun_position(df_merged["obs_date"])
	Az_sun = Az_sun - 180
	sunAboveHorizon = np.where(El_sun > 0, 1, 0)

	# Smallest angle between sun and pointing direction
	smallestAngleSunFirst = calcSmallestAngle(df_merged["ACTUALAZ_f"], df_merged["ACTUALEL_f"], Az_sun, El_sun)
	smallestAngleSunLast = calcSmallestAngle(df_merged["ACTUALAZ_l"], df_merged["ACTUALEL_l"], Az_sun, El_sun)
		
	# Wind direction compared to pointing direction
	df_merged["WINDDIRECTION_f_diff"] = df_merged["WINDDIRECTION_f"]-df_merged["ACTUALAZ_f"]
	df_merged["WINDDIRECTION_l_diff"] = df_merged["WINDDIRECTION_l"]-df_merged["ACTUALAZ_l"]

	#Scale columns to (-180,180) degree scale
	df_merged.loc[df_merged["WINDDIRECTION_f_diff"] > 180, "WINDDIRECTION_f_diff"] -= 360
	df_merged.loc[df_merged["WINDDIRECTION_f_diff"] < -180, "WINDDIRECTION_f_diff"] += 360
	df_merged.loc[df_merged["WINDDIRECTION_l_diff"] > 180, "WINDDIRECTION_l_diff"] -= 360
	df_merged.loc[df_merged["WINDDIRECTION_l_diff"] < -180, "WINDDIRECTION_l_diff"] += 360


	# Sun position relative to pointing, and smallest angle to sun
	df_merged["Az_sun"] = Az_sun
	df_merged["El_sun"] = El_sun
	df_merged["sunAboveHorizon"] = sunAboveHorizon
	df_merged["smallestAngleSunFirst"] = smallestAngleSunFirst
	df_merged["smallestAngleSunLast"] = smallestAngleSunLast
	df_merged["sunAz_diff_f"] = Az_sun - df_merged["ACTUALAZ_f"]
	df_merged["sunAz_diff_l"] = Az_sun - df_merged["ACTUALAZ_l"]
	df_merged["sunEl_diff_f"] = El_sun - df_merged["ACTUALEL_f"]
	df_merged["sunEl_diff_l"] = El_sun - df_merged["ACTUALEL_l"]

	#Scale columns to (-180,180) degree scale
	df_merged.loc[df_merged["sunAz_diff_f"] > 180, "sunAz_diff_f"] -= 360
	df_merged.loc[df_merged["sunAz_diff_f"] < -180, "sunAz_diff_f"] += 360
	df_merged.loc[df_merged["sunAz_diff_l"] > 180, "sunAz_diff_l"] -= 360
	df_merged.loc[df_merged["sunAz_diff_l"] < -180, "sunAz_diff_l"] += 360

	colOrder = df_merged.columns[9:12].tolist() + df_merged.columns[1:9].tolist() + df_merged.columns[12:].tolist()
	df_merged = df_merged.loc[:, colOrder]

	return df_merged



def removeOutliers(X_matrix):
    yAz = X_matrix[:,0].copy()
    yEl = X_matrix[:,1].copy()

    #Remove outliers
    outliersAz = np.where(np.abs(yAz) > np.mean(yAz) + 3*np.std(yAz))[0]
    outliersEl = np.where(np.abs(yEl) > np.mean(yEl) + 3*np.std(yEl))[0]
    outliers = np.unique(np.concatenate((outliersAz, outliersEl), axis=0))

    X_matrix = np.delete(X_matrix, outliers, axis=0)

    return X_matrix


def splitDf(df):
	X = df.loc[:, ~df.columns.isin(['Off_Az', 'Off_El', 'Off_eAz', 'Off_eEl', 'obs_date'])].copy()
	print(X.head())
	yAz = df.loc[:,"Off_Az"].copy()
	yEl = df.loc[:,"Off_El"].copy()

	return X, yAz, yEl

def removeOutliersDf(df, columns = ["Off_Az", "Off_El"]):
	#Remove outliers from data frame where colums are more than three standard deviations from the mean
	for col in columns:
		df = df[np.abs(df[col]-df[col].mean()) <= (3*df[col].std())]

	nanTest(df["Off_Az"])
	nanTest(df["Off_El"])

	return df


def runSimpleXGBoost(X,yAz,yEl):
    

    xgb_modelAz = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_seed)
    xgb_modelEl = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_seed)


    X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test = split_data(X, yAz, yEl, test_size=0.2, random_seed = random_seed)

    """
    #Scale data
    Scaler = StandardScaler()
    Scaler, X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test = scale_data(Scaler, X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test)
    """
    xgb_modelAz.fit(X_train, yAz_train)
    xgb_modelEl.fit(X_train, yEl_train)

    yAz_pred = xgb_modelAz.predict(X_test)
    yEl_pred = xgb_modelEl.predict(X_test)

    #Inverse scale
    """
    yAz_pred = Scaler.inverse_transform(yAz_pred.reshape(-1,1))
    yEl_pred = Scaler.inverse_transform(yEl_pred.reshape(-1,1))

    yAz_test = Scaler.inverse_transform(yAz_test.reshape(-1,1))
    yEl_test = Scaler.inverse_transform(yEl_test.reshape(-1,1))
    """
    mseAz = mean_squared_error(yAz_test, yAz_pred)
    mseEl = mean_squared_error(yEl_test, yEl_pred)

    print("Error Az: ", mseAz)
    print("Error El: ", mseEl)
    print("Error total: ", mseAz+mseEl)

    fig, ax = plt.subplots(figsize = (20,20))
    ax.scatter(yAz_test, yEl_test, color = "blue", label = "True")
    ax.scatter(yAz_pred, yEl_pred, color = "red", label = "Prediction")
    ax.set_xlabel("Az")
    ax.set_ylabel("El")
    ax.legend()
    fig.savefig(path_models + "AzEl_scatter_xgboost.png")
    plt.clf()

def plotCorrelations(X_df, name = ""):

    methods = ["pearson", "spearman"]

    for method in methods:
        plt.figure(figsize=(20,18))
        plt.title(f"{method.capitalize()} correlation between features and corrections")
        sns.heatmap(X_df.corr(method = method), annot=True, annot_kws={"size": 14})
        plt.tight_layout()
        plt.savefig(path_correlations + f"Correlation_{method}_{name}.png")
        plt.clf()

def plotFeatureImportance(model, name):

    fig, ax = plt.subplots(figsize = (20,20))
    xgb.plot_importance(model, ax = ax)#, tick_label = _cols)
    fig.savefig(path_features + f"Feature_importance_xgboost_{name}.png")
    plt.clf()

def plotFeatureImportance2(model, name):

    fig, ax = plt.subplots(figsize = (20,20))
    xgb.plot_importance(model, ax = ax, importance_type = "gain")
    fig.savefig(path_features + f"Feature_importance_xgboost_gain_{name}.png")
    plt.clf()

def plotFeatureImportance3(model, name):

    fig, ax = plt.subplots(figsize = (20,20))
    xgb.plot_importance(model, ax = ax, importance_type = "cover")
    fig.savefig(path_features + f"Feature_importance_xgboost_cover_{name}.png")
    plt.clf()

def plotFeatureImportance4(model, name):

    fig, ax = plt.subplots(figsize = (20,20))
    xgb.plot_importance(model, ax = ax, importance_type = "weight")
    fig.savefig(path_features + f"Feature_importance_xgboost_weight_{name}.png")
    plt.clf()

def plotFeatureImportance5(model, name):

    fig, ax = plt.subplots(figsize = (20,20))
    xgb.plot_importance(model, ax = ax, importance_type = "total_gain")
    fig.savefig(path_features + f"Feature_importance_xgboost_total_gain_{name}.png")
    plt.clf()

def plotFeatureImportance6(model, name):

    fig, ax = plt.subplots(figsize = (20,20))
    xgb.plot_importance(model, ax = ax, importance_type = "total_cover")
    fig.savefig(path_features + f"Feature_importance_xgboost_total_cover_{name}.png")
    plt.clf()

def plotAllFeatureImportance(modelAz, modelEl):

    plotFeatureImportance(modelAz, "Az")
    plotFeatureImportance2(modelAz, "Az")
    plotFeatureImportance3(modelAz, "Az")
    plotFeatureImportance4(modelAz, "Az")
    plotFeatureImportance5(modelAz, "Az")
    plotFeatureImportance6(modelAz, "Az")

    plotFeatureImportance(modelEl, "El")
    plotFeatureImportance2(modelEl, "El")
    plotFeatureImportance3(modelEl, "El")
    plotFeatureImportance4(modelEl, "El")
    plotFeatureImportance5(modelEl, "El")
    plotFeatureImportance6(modelEl, "El")


def plotPairs(X_df, targets = ["total_offset"]):

	cols = list(X_df.columns)
	print(cols)
	n_targets = 1
	n_feats = 6-n_targets

	for i in range(len(cols[2:]) // n_feats):
		plt.figure(figsize=(20,18))
		plt.title("Correlation between features and corrections")
		sns.pairplot(X_df, hue= "hour", vars=cols[:n_targets] + cols[n_feats*i+n_targets:n_feats*i+n_targets+n_feats], palette="BrBG")
		plt.savefig(path_correlations + f"Pairplot{i}.png",dpi = 400)
		plt.clf()

	#Remaining features
	if len(cols[n_targets:]) % n_feats != 0:
		last_i = len(cols[n_targets:]) // n_feats - 1
		plt.figure(figsize=(20,18))
		plt.title("Correlation between features and corrections")
		sns.pairplot(X_df, hue= "hour", vars=cols[:n_targets] + cols[n_feats*last_i+n_targets+n_feats:], palette="BrBG")
		plt.savefig(path_correlations + f"Pairplot{last_i+1}.png",dpi = 400)
		plt.clf()

	return


def objective(space):
    regr=xgb.XGBRegressor(
                    max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    X_train, X_test, y_train, y_test = space["data"]
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    regr.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=10,verbose=False)
    

    pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    #print ("MSE:", mse)
    return {'loss': mse, 'status': STATUS_OK }


def XGBoostTuned(df_merged, test=False):
	space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
			'gamma': hp.uniform ('gamma', 1,9),
			'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
			'reg_lambda' : hp.uniform('reg_lambda', 0,1),
			'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
			'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
			'seed': random_seed
		}


	X, yAz, yEl = splitDf(df_merged)
	print(X.shape, yAz.shape, yEl.shape)
	X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test = split_data(X, yAz, yEl, test_size=0.3, random_seed = random_seed)

	trials = Trials()
	space["data"] = (X_train, X_test, yAz_train, yAz_test)

	best_hyperparamsAz = fmin(fn = objective,
							space = space,
							algo = tpe.suggest,
							max_evals = 100,
							trials = trials)


	trials = Trials()
	space["data"] = (X_train, X_test, yEl_train, yEl_test)

	best_hyperparamsEl = fmin(fn = objective,
							space = space,
							algo = tpe.suggest,
							max_evals = 100,
							trials = trials)

	print(best_hyperparamsAz)
	print(best_hyperparamsEl)
	regrAz=xgb.XGBRegressor(
				max_depth = int(best_hyperparamsAz['max_depth']), gamma = best_hyperparamsAz['gamma'],
				reg_alpha = int(best_hyperparamsAz['reg_alpha']),min_child_weight=int(best_hyperparamsAz['min_child_weight']),
				colsample_bytree=int(best_hyperparamsAz['colsample_bytree']))

	regrEl=xgb.XGBRegressor(
					max_depth = int(best_hyperparamsEl['max_depth']), gamma = best_hyperparamsEl['gamma'],
					reg_alpha = int(best_hyperparamsEl['reg_alpha']),min_child_weight=int(best_hyperparamsEl['min_child_weight']),
					colsample_bytree=int(best_hyperparamsEl['colsample_bytree']))

	regrAz.fit(X_train, yAz_train)
	regrEl.fit(X_train, yEl_train)

	if test:
		testModel(X_test,yAz_test,yEl_test, regrAz, regrEl)


	return regrAz, regrEl

def testModel(X_test, yAz_test, yEl_test, modelAz, modelEl):
	yAz_pred = modelAz.predict(X_test)
	yEl_pred = modelEl.predict(X_test)

	mseAz = mean_squared_error(yAz_test,yAz_pred)
	mseEl = mean_squared_error(yEl_test, yEl_pred)

	print("Error Az: ", mseAz)
	print("Error El: ", mseEl)
	print("Error total: ", mseAz+mseEl)

	fig, ax = plt.subplots(figsize = (20,20))
	ax.scatter(yAz_test, yEl_test, color = "blue", label = "True")
	ax.scatter(yAz_pred, yEl_pred, color = "red", label = "Prediction")
	ax.set_xlabel("Az")
	ax.set_ylabel("El")
	ax.legend()
	fig.savefig(path_models + "AzEl_scatter_xgboost.png")
	plt.clf()

def nanTest(arr):
	if len(np.where(np.isnan(arr))[0]) > 0:
		print("Nan in array")
		return True

def SAGE(df_merged, paramsAz, paramsEl):

	X, yAz, yEl = splitDf(df_merged)
	_cols = X.columns

	test_len = 16
	X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test = split_data(X, yAz, yEl, test_size=0.3, random_seed = random_seed)
	X_train, X_test, yAz_train, yAz_test, yEl_train, yEl_test = X_train, X_test[:test_len], yAz_train, yAz_test[:test_len], yEl_train, yEl_test[:test_len]

	#XGB matrix
	dtrainAz = xgb.DMatrix(X_train, label=yAz_train)
	dvalAz = xgb.DMatrix(X_test, label=yAz_test)

	dtrainEl = xgb.DMatrix(X_train, label=yEl_train)
	dvalEl = xgb.DMatrix(X_test, label=yEl_test)

	evallistAz = [(dtrainAz, 'train'), (dvalAz, 'val')]
	evallistEl = [(dtrainEl, 'train'), (dvalEl, 'val')]

	num_round = 50

	modelAz = xgb.train(paramsAz, dtrainAz, num_round, evallistAz, verbose_eval=False)
	modelEl = xgb.train(paramsEl, dtrainEl, num_round, evallistEl, verbose_eval=False)

	regrAz.fit(X_train, yAz_train)
	regrEl.fit(X_train, yEl_train)
	imputerAz = sage.MarginalImputer(modelAz, X_test)
	estimatorAz = sage.PermutationEstimator(imputerAz, 'mse')
	sage_testuesAz = estimatorAz(X_test, yAz_test)
	sage_testuesAz.plot(feature_names=_cols)
	plt.savefig(path_features + "SAGE_xgboost_Az.png")

	imputerEl = sage.MarginalImputer(modelEl, X_test)
	estimatorEl = sage.PermutationEstimator(imputerEl, 'mse')
	sage_testuesEl = estimatorEl(X_test, yEl_test)
	sage_testuesEl.plot(feature_names=_cols)
	plt.savefig(path_features + "SAGE_xgboost_El.png")

def calcSmallestAngle(A,B,C,D):
    A = np.deg2rad(A)
    B = np.deg2rad(B)
    C = np.deg2rad(C)
    D = np.deg2rad(D)

    #Calculating angle
    dotprod = np.cos(A)*np.sin(B)*np.cos(C)*np.sin(D) + np.sin(A)*np.sin(B)*np.sin(C)*np.sin(D) + np.cos(B)*np.cos(D)
    angle = np.arccos(dotprod)

    return np.rad2deg(angle)
