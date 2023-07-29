import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


random_seed = 12
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
sns.set(font_scale = 2)
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18

pt = pd.read_csv("PointingTable.csv")
pt["obs_date"] = pd.to_datetime(pt["obs_date"])

ms = pd.read_csv("meteoscope.csv")
"""
Meteoscope.csv parameters
-3: Dewpoint [C] | 
-2: Air pressure hPa
0: Temperature [K]
1: Relative humidity [%]
5: Wind speed [m/s]
6: Wind direction [deg]
12: Precipitable water vapour [mm]
"""

params = {
	-3: {
			"name": "Dewpoint",
			"expected_min": None,
			"expected_max": None
	},
	-2: {
			"name": "Air pressure",
			"expected_min": 200,
			"expected_max": 2000
	},
	0: {
			"name": "Temperature",
			"expected_min": 180-273.15,
			"expected_max": 320-273.15
	},
	1: {
			"name": "Relative humidity",
			"expected_min": 0,
			"expected_max": 100
	},
	5: {
			"name": "Wind speed",
			"expected_min": 0,
			"expected_max": 150
	},
	6: {
			"name": "Wind direction",
			"expected_min": 0,
			"expected_max": 360
	},
	12: {
			"name": "Precipitable water vapour",
			"expected_min": 0,
			"expected_max": 50
	}
}



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

ms_joined = make_df(ms)

Az = np.array(pt["Az"])
El = np.array(pt["El"])

y_Az = np.array(pt["Az"] + pt["Off_Az"])
y_El = np.array(pt["El"] + pt["Off_El"])

off_Az = np.array(pt["Off_Az"])
off_El = np.array(pt["Off_El"])

"""
Finds the data that is closest to in time
pt: PointingTable
ms: Meteoscope
tm: Tiltmeter
"""
#find the data in the different dataframe that are the closest in time to the pointing scan

#Time of pointing scan
time_ps = np.array(pt["obs_date"]).astype("datetime64[ms]")
time_ms = np.array(ms_joined["date_measured"]).astype("datetime64[ms]")

diff = time_ps[:, np.newaxis] - time_ms[np.newaxis]
diff = np.where(diff >= np.timedelta64(0, 'ms'), diff, 1e8)
diff_min = np.amin(diff, axis=1)
#print(diff_min, type(diff_min[0]))
#print(np.where(diff_min > np.timedelta64(1800000, 'ms')))
#Contains index for data in meteoscope data that is the closest in time to pointing scan.
idx_min = np.array([np.where(diff[i] == diff_min[i])[0][0] for i in range(len(diff))])
#print(time_ps[200], time_ms[idx_min[200]])
#print(idx_min)
#print(ms_joined.head())



columns = ['median_dewpoint', 'mean_dewpoint',
       'first_dewpoint', 'median_air_pressure',
       'mean_air_pressure', 'first_air_pressure', 'median_temperature',
       'mean_temperature', 'first_temperature', 'median_relative_humidity',
       'mean_relative_humidity', 'first_relative_humidity',
       'median_wind_speed', 'mean_wind_speed', 'first_wind_speed',
       'median_wind_direction', 'mean_wind_direction', 'first_wind_direction',
       'median_pwv', 'mean_pwv', 'first_pwv']

"""ap = ms_joined["mean_air_pressure"][(ms_joined["mean_air_pressure"] > 200) & (ms_joined["mean_air_pressure"] < 2000)]
dp = ms_joined["mean_dewpoint"][(ms_joined["mean_dewpoint"] > -100)]
pwv = ms_joined["mean_pwv"][(ms_joined["mean_pwv"] > 0) & ms_joined["mean_pwv"] < 50]
rh = ms_joined["mean_relative_humidity"][(ms_joined["mean_relative_humidity"] > 0) & (ms_joined["mean_relative_humidity"] < 100)]
t = ms_joined["mean_temperature"][(ms_joined["mean_temperature"] > -100) & (ms_joined["mean_temperature"] < 50)]
wd = ms_joined["mean_wind_direction"][(ms_joined["mean_wind_direction"] > 0) & (ms_joined["mean_wind_direction"] < 360)]
ws = ms_joined["mean_wind_speed"][(ms_joined["mean_wind_speed"] > 0) & (ms_joined["mean_wind_speed"] < 150)]"""

ms_joined_obs = ms_joined.iloc[idx_min]


def data_hist(df):
	cols = ["median", "mean", "first"]
	names = ["dewpoint", "air_pressure", "temperature", "relative_humidity", "wind_speed", "wind_direction", "pwv"]
	
	for i,name in enumerate(names):
		fig, axs = plt.subplots(3, figsize = (20,18))
		for j,col in enumerate(cols):
			df[col + "_" + name] = df[col + "_" + name].hist(ax = axs[j])
			axs[j].set_title(col)
		fig.savefig(name + "_obs_hists.png")
		plt.clf()	

mean_median_pwv = ms_joined_obs["median_pwv"][(ms_joined_obs["median_pwv"] > 0) & (ms_joined_obs["median_pwv"] < 50)].mean()
ms_joined_obs["median_pwv"] = np.where(ms_joined_obs["median_pwv"] < 0, mean_median_pwv, ms_joined_obs["median_pwv"])

X = np.array([
	pt["Az"],
	pt["El"],
	ms_joined_obs["mean_dewpoint"],
	ms_joined_obs["mean_air_pressure"],
	ms_joined_obs["mean_temperature"],
	ms_joined_obs["mean_relative_humidity"],
	ms_joined_obs["mean_wind_speed"],
	ms_joined_obs["mean_wind_direction"],
	ms_joined_obs["median_pwv"]
	]).T

print(X.shape)

X_train, X_test, yAzTrain, yAzTest = train_test_split(X, off_Az, random_state=random_seed, test_size=0.3)
X_train, X_test, yElTrain, yElTest = train_test_split(X, off_El, random_state=random_seed, test_size=0.3)

def linreg_test(X_train, X_test, yAzTrain, yAzTest, yElTrain, yElTest):
	print(X_train.shape, X_test.shape, yAzTrain.shape, yAzTest.shape, yElTrain.shape, yElTest.shape)

	modelAz = LinearRegression()
	modelEl = LinearRegression()

	modelAz.fit(X_train, yAzTrain)
	modelEl.fit(X_train, yElTrain)

	predAz = modelAz.predict(X_test)
	predEl = modelEl.predict(X_test)

	predtrainAz = modelAz.predict(X_train)
	predtrainEl = modelEl.predict(X_train)

	eAz = np.sqrt(np.mean((predAz - yAzTest)**2))
	eEl = np.sqrt(np.mean((predEl - yElTest)**2))

	eAztrain = np.sqrt(np.mean((predtrainAz - yAzTrain)**2))
	eEltrain = np.sqrt(np.mean((predtrainEl - yElTrain)**2))

	print("Az error:", eAz)
	print("El error:", eEl)

	print("Az train error:", eAztrain)
	print("El train error:", eEltrain)

	#make 2d scatterplot with Az on x-axis and El on y-axis. Plot off_Az and off_El as blue, and predAz and predEl as red.
	fig, ax = plt.subplots(figsize = (20,20))
	ax.scatter(yAzTest, yElTest, color = "blue", label = "Test")
	ax.scatter(predAz, predEl, color = "red", label = "Prediction")
	ax.set_xlabel("Az")
	ax.set_ylabel("El")
	ax.legend()
	fig.savefig("AzEl_scatter_linreg.png")
	plt.clf()

#linreg_test(X_train, X_test, yAzTrain, yAzTest, yElTrain, yElTest)

def nn_test(X_train, X_test, yAzTrain, yAzTest, yElTrain, yElTest):
	regrAz = MLPRegressor(random_state=random_seed, max_iter=500, learning_rate_init= 0.0001,  hidden_layer_sizes=(60,100,100,50)).fit(X_train, yAzTrain)
	regrEl = MLPRegressor(random_state=random_seed, max_iter=500, learning_rate_init= 0.0001,  hidden_layer_sizes=(60,100,100,50)).fit(X_train, yElTrain)

	predAz = regrAz.predict(X_test)
	predEl = regrEl.predict(X_test)

	predtrainAz = regrAz.predict(X_train)
	predtrainEl = regrEl.predict(X_train)

	eAz = np.sqrt(np.mean((predAz - yAzTest)**2))
	eEl = np.sqrt(np.mean((predEl - yElTest)**2))

	eAztrain = np.sqrt(np.mean((predtrainAz - yAzTrain)**2))
	eEltrain = np.sqrt(np.mean((predtrainEl - yElTrain)**2))

	print("Az error:", eAz)
	print("El error:", eEl)

	print("Az train error:", eAztrain)
	print("El train error:", eEltrain)

	fig, ax = plt.subplots(figsize = (20,20))
	ax.scatter(yAzTest, yElTest, color = "blue", label = "Test")
	ax.scatter(predAz, predEl, color = "red", label = "Prediction")
	ax.set_xlabel("Az")
	ax.set_ylabel("El")
	ax.legend()
	fig.savefig("AzEl_scatter_nn.png")
	plt.clf()

linreg_test(X_train, X_test, yAzTrain, yAzTest, yElTrain, yElTest)
nn_test(X_train, X_test, yAzTrain, yAzTest, yElTrain, yElTest)