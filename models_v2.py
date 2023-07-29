import importlib
import xgboost as xgb
import lightgbm as lgb
from dataset import PrepareData, PointingScansClassification, PointingScansClassification_v2,PrepareDataAnalytical, PrepareDataAnalytical_v2, PrepareDataAnalytical_v3
from settings import patches, features, dataset_params
from hyperopt import hp
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from IPython import embed
import pandas as pd
import json
import sage
import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error,make_scorer, log_loss
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
random_seed = 412069413
MAX_EVALS = 200

# sns.set(font_scale=1.5)

plt.rcParams['axes.titlesize'] = 18; plt.rcParams['axes.labelsize'] = 18;
plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18


class Model():
    """
    Parent class for all model classes
    Contains common functions like plotting, evaluation,
    and feature importance

    All childclasses need to implement these attributes:
     - dataset
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.params  = dataset.params 
        self.X_train, self.X_test, self.y_train, self.y_test = dataset.get_data()
    

        if self.params is not None:
            if self.params['new_model']:
                self.PATH_MODEL = f'./NewModel/Models/' 
                self.PATH_PLOTS = f'./NewModel/Plots/' 

            elif self.params['optical_model']:
                self.PATH_MODEL = './AnalyticalModelRaw/Models/'
                self.PATH_PLOTS = './AnalyticalModelRaw/Plots/'
              
            
            else:
                self.PATH_MODEL = f'./Results_v3/{dataset.params["patch_key"]}/{dataset.params["feature_key"]}/Models/' 
                self.PATH_PLOTS = f'./Results_v3/{dataset.params["patch_key"]}/{dataset.params["feature_key"]}/Plots/'
                print('Feature key', dataset.params['feature_key'])

        else:
            self.PATH_MODEL = f'./FinalResults/Run{dataset.run_number}/Models/'
            self.PATH_PLOTS = f'./FinalResults/Run{dataset.run_number}/Plots/'

    def get_name(self):
        return NotImplementedError('Must be implented in child model class')

    def plot_sorted_predictions_v2(self):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)


        if self.dataset.target_key == 'testaz':
            pred = self.model.predict(self.X_test)-self.X_test[:,0]
            real = self.y_test - self.X_test[:,0]
        elif self.dataset.target_key == 'testel':
            pred = self.model.predict(self.X_test)-self.X_test[:,1]
            real = self.y_test - self.X_test[:,1]

        pred = np.rad2deg(pred) * 3600
        real = np.rad2deg(real) * 3600
        

        me = mean_absolute_error(real, pred)
        
        embed()

        if self.dataset.target == 'both':
            noPred = np.mean(np.sqrt(self.dataset.df['Off_Az']**2 + self.dataset.df['Off_El']**2))
        else:
            try:
                noPred = np.mean(np.abs(real))
            except:
                embed()
            


        plt.clf()
        plt.figure(figsize=(18,12))


        if self.dataset.n_targets > 1:
            idxSorted0 = self.y_test[:,0].argsort()
            idxSorted1 = self.y_test[:,1].argsort()
        
            plt.plot(range(len(pred)), pred[idxSorted0,0], label="Prediction Az")
            plt.plot(range(len(pred)), pred[idxSorted1,1], label="Prediction El")
            
            
            plt.plot(range(len(pred)), self.y_test[idxSorted0, 0], label='Real Az')
            plt.plot(range(len(pred)), self.y_test[idxSorted1, 1], label='Real El')

        else:
            print('y_test shape: ', self.y_test.shape)
            try:
                idxSorted = real.ravel().argsort()
            except:
                idxSorted = real.argsort()
            plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
            try:
                plt.plot(range(len(pred)), real[idxSorted], label="Real data")
            except:
                embed()

        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | ME: {me:.3f} | {noPred:.3f}")
        plt.legend()

        if self.params is not None:
            if self.params['use_pca'] is True:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_pca_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_pca_{self.params['target']}.png")
            else:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_{self.params['target']}.png")
        else:
            save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}.png")
            save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
    def plot_sorted_predictions(self):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)


        
        pred = self.model.predict(self.X_test)
        me = mean_absolute_error(self.y_test, pred)
        
        if self.dataset.target == 'both':
            noPred = np.mean(np.sqrt(self.dataset.df['Off_Az']**2 + self.dataset.df['Off_El']**2))
        else:
            try:
                noPred = self.dataset.df.loc[:, self.dataset.target].abs().values.mean() 
            except:
                embed()
            


        plt.clf()
        plt.figure(figsize=(18,12))


        if self.dataset.n_targets > 1:
            idxSorted0 = self.y_test[:,0].argsort()
            idxSorted1 = self.y_test[:,1].argsort()
        
            plt.plot(range(len(pred)), pred[idxSorted0,0], label="Prediction Az")
            plt.plot(range(len(pred)), pred[idxSorted1,1], label="Prediction El")
            
            
            plt.plot(range(len(pred)), self.y_test[idxSorted0, 0], label='Real Az')
            plt.plot(range(len(pred)), self.y_test[idxSorted1, 1], label='Real El')

        else:
            print('y_test shape: ', self.y_test.shape)
            try:
                idxSorted = self.y_test.ravel().argsort()
            except:
                idxSorted = self.y_test.argsort()
            plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
            try:
                idxSorted = self.y_test.ravel().argsort()
                plt.plot(range(len(pred)), self.y_test.ravel()[idxSorted], label="Real data")
            except:
                embed()


        me = np.rad2deg(me) * 3600

        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | ME: {me:.3f} | {noPred:.3f}")
        plt.legend()

        if self.params is not None:
            if self.params['use_pca'] is True:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_pca_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_pca_{self.params['target']}.png")
            else:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_{self.params['target']}.png")
        else:
            save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}.png")
            save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
    

    def plot_sorted_prediction_v2(self,elevation_for_scaling=None):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)


        
        pred = self.model.predict(self.X_test)
        residual = pred - self.y_test
        if elevation_for_scaling is not None:
            residual *= np.cos(elevation_for_scaling)
        
        RMS = np.sqrt(np.mean( np.rad2deg(residual * 3600)**2 ) )


        plt.clf()
        plt.figure(figsize=(18,12))


        print('y_test shape: ', self.y_test.shape)
        try:
            idxSorted = self.y_test.ravel().argsort()
        except:
            idxSorted = self.y_test.argsort()
        plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
        try:
            idxSorted = self.y_test.ravel().argsort()
            plt.plot(range(len(pred)), self.y_test.ravel()[idxSorted], label="Real data")
        except:
            embed()


        # me = np.rad2deg(RMS) * 3600
        me = RMS
        print(f'RMS for {self.name}: {me} arcsecs')
        benchmark = self.dataset.benchmark
        nopred = np.mean(np.abs(self.y_test))

        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | RMS: {me:.3f}\'\' | With corrections: {benchmark:.3f}\'\' | Optical Model: {nopred:.3f}\'\'")
        plt.legend()

        save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()

    def plot_sorted_prediction_final(self, X=None, y=None, fn=''):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')

        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        pred = self.model.predict(X)
        residual = pred - y        
        RMS = np.sqrt( np.mean( residual**2 ) )


        plt.clf()
        plt.figure(figsize=(12,8))


        print('y_test shape: ', y.shape)
        try:
            idxSorted = y.ravel().argsort()
        except:
            idxSorted = y.argsort()
        plt.plot(range(len(pred)), pred[idxSorted], label="Predicted")
        try:
            idxSorted = y.ravel().argsort()
            plt.plot(range(len(pred)), y.ravel()[idxSorted], label="True")
        except:
            embed()

        print(f'RMS for {self.name}: {RMS} arcsecs')
        rms_offset = self.dataset.rms_offset
        rms_offset_optimal_correction = self.dataset.rms_offset_optimal_correction


        plt.xlabel("Sample #")
        plt.ylabel("Offset [\'\']")
        # plt.title(f"{self.name} | RMS: {RMS:.2f}\'\' | With corrections: {rms_offset:.2f}\'\' | Optimal model: {rms_offset_optimal_correction:.2f}\'\'")
        plt.title(f'True and predicted offset')
        plt.legend()

        save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}{fn}.pdf")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
        return RMS, rms_offset

    def plot_histogram(self, X=None, y=None, fn=''):
        print(f"Plotting histogram for {self.name}")
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_HISTOGRAM):
            os.makedirs(PATH_HISTOGRAM)

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        pred = self.model.predict(X)

        plt.clf()

        n_bins = 25
        plt.figure(figsize=(12,8))
        #Increase title and lable size
    
        _, bins, _ = plt.hist(y, bins = n_bins, alpha = 0.8, label = 'Current offset')
        plt.hist(y - pred, bins = bins, alpha = 0.8, label='XGB Model')
        plt.xlabel('Offset [\'\']')#, fontsize = fontsize)
        plt.ylabel('Number of samples')#, fontsize = fontsize)
        # plt.xticks(fontsize = fontsize)
        # plt.yticks(fontsize = fontsize)
        plt.title('Distribution of pointing offsets with and without XGB model')
        #plt.legend(fontsize = fontsize)
        plt.legend()
        plt.tight_layout()

        save_path_hist = os.path.join(PATH_HISTOGRAM, f"hist_{self.name}{fn}.pdf")

        plt.savefig(save_path_hist, dpi = 400)


    def plot_error_locations(self):
        print(f"Plotting error locations for {self.name}")
        PATH_ERRORLOC = os.path.join(self.PATH_PLOTS, f'ErrorLocations/')
        if not os.path.exists(PATH_ERRORLOC):
            os.makedirs(PATH_ERRORLOC)


        pred = self.model.predict(self.X_test)
        # Calculate the errors between the true y and predicted y

        errors = np.sqrt((pred - self.y_test)**2)

        # Create a scatter plot of the true y and predicted y
        fig = plt.figure() 
        ax = fig.add_subplot()
        ax.scatter(self.X_train[:,0], self.X_train[:,1], c='blue', label='Training data')

        # Mark the data points with large errors in a different color
        threshold = np.percentile(errors, 90)
        threshold = np.deg2rad(2/3600)
        indices = np.where(errors > threshold)[0]
        print(f'Number of samples with large errors: {len(indices)}, threshold: {threshold:.3f}')

        ax.scatter(self.X_test[indices,0],self.X_test[indices,1], c='red', label='Large error on test data')

        ax.set_xlabel('Azimuth')
        ax.set_ylabel('Elevation')
        plt.legend()

        if self.params is not None:
            if self.params['use_pca'] is True:
                save_path_errloc = os.path.join(PATH_ERRORLOC,  f"errloc_{self.name}_pca_{self.params['target']}.png")
            else:
                save_path_errloc = os.path.join(PATH_ERRORLOC,  f"errloc_{self.name}_{self.params['target']}.png")
        else:
            save_path_errloc = os.path.join(PATH_ERRORLOC,  f"errloc_{self.name}.png")

        plt.savefig(save_path_errloc, dpi = 400)
        plt.clf()
        return

"""
errors = np.abs(self.y_test - pred)

# Create a scatter plot of the true y and predicted y
plt.clf()
fig = plt.figure()
ax = fig.add_subplot()
# ax.scatter(y[:, 0], y[:, 1], c='blue', label='True y')
# ax.scatter(y_pred[:, 0], y_pred[:, 1], c='green', label='Predicted y')

# Mark the data points with large errors in a different color
threshold = 1
indices = np.where(errors > threshold)[0]
ax.scatter(np.rad2deg(self.X_train[:,0]), np.rad2deg(self.X_train[:,1]), c='blue', label='Training data')
ax.scatter(np.rad2deg(self.X_test[indices, 0]), np.rad2deg(self.X_test[indices, 1]), c='red', label='Large errors')
plt.title('Scatter plot showing location of training')
plt.xlabel('Azimuth')
plt.ylabel('Elevation')
plt.legend()
plt.savefig('./AnalyticalModel/test_scatter_errors.png')
"""

class XGBoostRegressor(Model):
    def __init__(self, dataset, name = 'XGB', load_model = False, train_model = False, model_path = None):
        super().__init__(dataset)

    
        self.name = name



        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest  = xgb.DMatrix(self.X_test, label=self.y_test)

        if not isinstance(self.X_train, type(self.y_train)):
            self.y_train = self.y_train.values.ravel()
            self.y_test  = self.y_test.values.ravel()

        else:
            self.y_train = self.y_train.values.ravel()
            self.y_test  = self.y_test.values.ravel()
            self.X_train = self.X_train.values
            self.X_test  = self.X_test.values

        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)

        if load_model:
            self.load_model(model_path)

        if train_model:
            self.train()


    def train(self, save = True):
        space={'max_depth': hp.quniform("max_depth", 1, 5, 1),
                # 'gamma': hp.uniform ('gamma', 1,9),
                # 'reg_alpha' : hp.quniform('reg_alpha', 10,180,1),
                'reg_lambda' : hp.uniform('reg_lambda', 0,1),
                'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
                # 'min_child_weight' : hp.quniform('min_child_weight', 0, 100, 1),
                'n_estimators': hp.quniform('n_estimators', 20,500,1),
            }
        
        space={'max_depth': hp.quniform("max_depth", 1, 5, 1),
                'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'n_estimators': hp.quniform('n_estimators', 20, 500, 1),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'gamma': hp.loguniform('gamma', -5, 0),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            }

        evaluation = [(self.dtest, 'eval'), (self.dtrain, 'train')]

        #search = xgb.train(params, self.dtrain, 1, evals = evaluation, verbose_eval = True)

        trials = Trials()
        self.model_params = fmin(fn = self.objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = MAX_EVALS,
                            trials = trials)

        print(f'-----------------------params-----------------------')
        print(self.model_params)

        self.model_params['max_depth'] = int(self.model_params['max_depth'])
        self.model_params['n_estimators'] = int(self.model_params['n_estimators'])
        # self.model_params['min_child_weight'] = int(self.model_params['min_child_weight'])
        try:
            self.model = xgb.XGBRegressor(**self.model_params, n_jobs = 24)
            self.model.fit(self.X_train,self.y_train)
        except:
            embed()
        #print("Real MSE:", (self.y_test**2).mean())
        #self.model = self.model.get_booster()

        if save:
            self.save_model()
        
        return self.model_params




    def objective(self, space):
        #regr=xgb.XGBRegressor(max_depth = space['max_depth'], n_estimators = int(space['n_estimators']))
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        # print('-------------', space)
        space['n_estimators'] = int(space['n_estimators'])
        space['max_depth'] = int(space['max_depth'])
        # space['min_child_weight'] = int(space['min_child_weight'])
        #model = xgb.train(space, self.dtrain, space['num_rounds'], evals = evaluation, verbose_eval = True)
        regr = xgb.XGBRegressor(**space, n_jobs = 24)
        regr.fit(self.X_train, self.y_train, eval_set=evaluation,verbose=False)
        

        pred = regr.predict(self.X_test)
        mse = mean_squared_error(self.y_test, pred)
        #print ("MSE:", mse)
        return {'loss': mse, 'status': STATUS_OK }


    def save_model(self):
        #save model with pickle
        if not os.path.exists(self.PATH_MODEL):
            os.makedirs(self.PATH_MODEL)
        print('Saving model')
        path = os.path.join(self.PATH_MODEL, f'{self.name}.pkl')
        pickle.dump(self.model, open(path, 'wb'))


    def load_model(self, model_path):
        if model_path is not None:
            path = model_path
        else:
            path = os.path.join(self.PATH_MODEL, f'{self.name}.pkl')

        if os.path.exists(path):
            self.model = pickle.load(open(path, 'rb'))
        else:
            print('No model found')

class XGBoostClassifier(Model):
    def __init__(self, dataset, load_model = False, train_model = False):
        # super().__init__(dataset)

        self.name = 'XGB'
        self.threshold = 0.5
        self.dataset = dataset
        self.X_train, self.X_test, self.y_train, self.y_test = dataset.get_data()

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest  = xgb.DMatrix(self.X_test, label=self.y_test)

        self.neg_pos_ratio = self.dataset.get_neg_over_pos()
        print(self.neg_pos_ratio, sum(self.y_train == 1), sum(self.y_train == 0))
        # self.X_train = self.X_train.values
        # self.X_test  = self.X_test.values
        # self.y_train = self.y_train.values
        # self.y_test  = self.y_test.values

        if load_model:
            self.load_model()

        if train_model:
            self.train()


        # Make predictions on the test set
        y_pred = self.predict_with_threshold(self.threshold)

        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        # Extract true positives, false positives, true negatives, and false negatives
        tp = cm[1, 1]
        fp = cm[0, 1]
        tn = cm[0, 0]
        fn = cm[1, 0]

        # Compute false positive rate and false negative rate
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        # Print results
        print(f"Accuracy: {accuracy:.2f}")
        print(f"False positive rate: {fpr:.2f}")
        print(f"False negative rate: {fnr:.2f}")

        

        


        # Calculate precision and recall and plot the curve
        # plt.figure(figsize=(12,8))
        precision, recall, _ = precision_recall_curve(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-recall curve')
        plt.tight_layout()
        plt.savefig(f'./ClassifyingScans_v2/precision_recall_curve_{dataset.var}.png', dpi = 300)
        plt.clf()

    

        y_pred_proba = self.model.predict_proba(self.X_test)
        # Calculate the average precision for each threshold
        thresholds = np.linspace(0, 1, 101)
        average_precisions = []
        for threshold in thresholds:
            y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
            average_precisions.append(average_precision_score(self.y_test, y_pred))

        # Plot the mAP curve
        # plt.figure(figsize=(8,6))
        plt.plot(thresholds, average_precisions)
        plt.xlabel('Threshold')
        plt.ylabel('Average precision')
        plt.title('Average precision for each classification threshold')
        plt.tight_layout()
        plt.savefig(f'./ClassifyingScans_v2/mAP_curve_{dataset.var}.png', dpi = 300)
        plt.clf()
        
        print(f'Max average precision: {max(average_precisions):.2f} with threshold(s) {thresholds[np.argmax(average_precisions)]}')

    def train(self, save = True):
        print('Training XGBoost Classifier')
        space={
            'max_depth':  hp.choice('max_depth', np.arange(3, 15, dtype=int)),
            'num_rounds': hp.choice('num_rounds', np.arange(100, 1000, dtype=int)),
            'min_child_samples': hp.choice('min_child_samples', [5,10,15,20]),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'num_leaves': hp.choice('num_leaves', np.arange(10, 50, dtype=int)),
            'feature_pre_filter': hp.choice('feature_pre_filter', [False]),
            'seed'   : random_seed
            }
        
        space={'max_depth': hp.quniform("max_depth", 1, 5, 1),
                # 'gamma': hp.uniform ('gamma', 1,9),
                # 'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
                # 'reg_lambda' : hp.uniform('reg_lambda', 0,1),
                # 'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
                # 'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
                # 'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'n_estimators': hp.quniform('n_estimators', 1,80,1),
                # 'scale_pos_weight': hp.choice('scale_pos_weight', [self.neg_pos_ratio])
            }
        """
        space = {
            'max_depth': hp.quniform('max_depth', 0, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
        }
        """

        hyperparameters = {
            'colsample_bynode': 0.8,
            'learning_rate': 1,
            'max_depth': 10,
            'num_parallel_tree': 500,
            'subsample': 0.8
            }

        evaluation = [(self.dtest, 'eval'), (self.dtrain, 'train')]

        #search = xgb.train(params, self.dtrain, 1, evals = evaluation, verbose_eval = True)

        trials = Trials()
        self.model_params = fmin(fn = self.objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = MAX_EVALS,
                            trials = trials)

        print(f'-----------------------parrams-----------------------')
        self.model_params['max_depth'] = int(self.model_params['max_depth'])
        self.model_params['n_estimators'] = int(self.model_params['n_estimators'])

        try:
            self.model = xgb.XGBClassifier(**self.model_params, scale_pos_weight = self.neg_pos_ratio)
            self.model.fit(self.X_train,self.y_train)
        except:
            embed()
        #print("Real MSE:", (self.y_test**2).mean())
        #self.model = self.model.get_booster()


        if save:
            self.save_model()

        return self.model_params

    def objective(self, space):
        #regr=xgb.XGBRegressor(max_depth = space['max_depth'], n_estimators = int(space['n_estimators']))
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        # print('-------------', space)
        space['n_estimators'] = int(space['n_estimators'])
        space['max_depth'] = int(space['max_depth'])
        #model = xgb.train(space, self.dtrain, space['num_rounds'], evals = evaluation, verbose_eval = True)
        clf = xgb.XGBClassifier(**space, n_jobs = 16, scale_pos_weight = self.neg_pos_ratio)
        clf.fit(self.X_train, self.y_train, eval_set=evaluation,verbose=False)
        #print clf parameters
        # print(clf.get_params()['max_depth'],clf.get_params()['n_estimators'])
        pred = self.predict_with_threshold(model = clf)
        
        # pred = regr.predict(self.X_test)
        loss = log_loss(self.y_test, pred)
        #print ("MSE:", mse)
        return {'loss': loss, 'status': STATUS_OK }

    def predict_with_threshold(self, threshold = 0.5, model = None):
        if model is not None:
            probs = model.predict_proba(self.X_test)
        else:
            probs = self.model.predict_proba(self.X_test)

        pred = np.where(probs[:,1] > threshold, 1, 0)

        return pred

    def save_model(self):
        #save model with pickle
        print('Saving model')
        pickle.dump(self.model, open(f'./ClassifyingScans_v2/Models/{self.name}_{self.dataset.var}.pkl', 'wb'))

    def load_model(self):
        if os.path.exists(f'./ClassifyingScans_v2/Models/{self.name}_{self.dataset.var}.pkl'):
            self.model = pickle.load(open(f'./ClassifyingScans_v2/Models/{self.name}_{self.dataset.var}.pkl', 'rb'))
        else:
            print('No model found')

class RandomForest(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.name = 'RF'



        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest  = xgb.DMatrix(self.X_test, label=self.y_test)

        self.X_train = self.X_train.values
        self.X_test  = self.X_test.values
        self.y_train = self.y_train.values
        self.y_test  = self.y_test.values

        self.evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

        self.train()


    def train(self, save = True):
        space = {
            'max_depth': hp.quniform("max_depth", 1, 18, 1),
            # 'n_estimators': hp.quniform('n_estimators', 100,500,1),
            # 'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 0.8),
            # 'learning_rate': hp.choice('learning_rate', [1]),
            # 'subsample': hp.uniform('subsample', 0.5, 0.8),
            'n_estimators': hp.quniform('n_estimators', 100, 800, 1),

        }
        
        params = {
        'colsample_bynode': 0.8,
        'learning_rate': 1,
        'max_depth': 5,
        'num_parallel_tree': 100,
        'objective': 'binary:logistic',
        'subsample': 0.8,
        'tree_method': 'gpu_hist'
        }

        evaluation = [(self.dtest, 'eval'), (self.dtrain, 'train')]

        #search = xgb.train(params, self.dtrain, 1, evals = evaluation, verbose_eval = True)

        trials = Trials()
        self.model_params = fmin(fn = self.objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = MAX_EVALS,
                            trials = trials)

        print(f'-----------------------parrams-----------------------')
        print(self.model_params)
        self.model_params['max_depth'] = int(self.model_params['max_depth'])
        self.model_params['n_estimators'] = int(self.model_params['n_estimators'])

        try:
            self.model = RandomForestRegressor(**self.model_params)
            self.model.fit(self.X_train,self.y_train)
        except:
            embed()
        #print("Real MSE:", (self.y_test**2).mean())
        #self.model = self.model.get_booster()

        return
        if save:
            self.save_model()

    def objective(self, space):
        #regr=xgb.XGBRegressor(max_depth = space['max_depth'], n_estimators = int(space['n_estimators']))
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        print('-------------', space)
        space['n_estimators'] = int(space['n_estimators'])
        space['max_depth'] = int(space['max_depth'])
        #model = xgb.train(space, self.dtrain, space['num_rounds'], evals = evaluation, verbose_eval = True)
        regr = RandomForestRegressor(**space)
        regr.fit(self.X_train, self.y_train, eval_set=evaluation,verbose=False)
        

        pred = regr.predict(self.X_test)
        mse = mean_squared_error(self.y_test, pred)
        #print ("MSE:", mse)
        return {'loss': mse, 'status': STATUS_OK }

class LightGBM(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.name = 'LGB'



        self.dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        self.dtest  = lgb.Dataset(self.X_test, label=self.y_test)

        
        if not isinstance(self.X_train, type(self.y_train)):
            self.y_train = self.y_train.values.ravel()
            self.y_test  = self.y_test.values.ravel()

        else:
            self.y_train = self.y_train.values.ravel()
            self.y_test  = self.y_test.values.ravel()
            self.X_train = self.X_train.values
            self.X_test  = self.X_test.values

        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)
        self.train()



    def train(self, save = True):
        space={
            'max_depth':  hp.choice('max_depth', np.arange(3, 15, dtype=int)),
            'num_rounds': hp.choice('num_rounds', np.arange(100, 1000, dtype=int)),
            'min_child_samples': hp.choice('min_child_samples', [5,10,15,20]),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'num_leaves': hp.choice('num_leaves', np.arange(10, 50, dtype=int)),
            'feature_pre_filter': hp.choice('feature_pre_filter', [False]),
        
            'seed'   : random_seed
            }
        space = {
            'max_depth': hp.quniform("max_depth", 1, 10, 1),
            'n_estimators': hp.quniform('n_estimators', 25,250,1),
            #'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
            #'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            #'min_child_samples': hp.quniform('min_child_samples', 4,20,1),
            'seed': random_seed
        }

        #search = xgb.train(params, self.dtrain, 1, evals = evaluation, verbose_eval = True)

        trials = Trials()
        self.model_params = fmin(fn = self.objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = MAX_EVALS,
                            trials = trials)

        print(self.model_params)

        self.model_params['max_depth'] = int(self.model_params['max_depth'])
        self.model_params['n_estimators'] = int(self.model_params['n_estimators'])
        #self.model_params['min_child_samples'] = int(self.model_params['min_child_samples'])

        try:
            self.model = lgb.LGBMRegressor(**self.model_params)
            self.model.fit(self.X_train,self.y_train)
        except:
            embed()
        #print("Real MSE:", (self.y_test**2).mean())
        #self.model = self.model.get_booster()

        return
        if save:
            self.save_model()




    def objective(self, space):
        #regr=xgb.XGBRegressor(max_depth = space['max_depth'], n_estimators = int(space['n_estimators']))
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

        
        print('-------------', space)
        space['max_depth'] = int(space['max_depth'])
        space['n_estimators'] = int(space['n_estimators'])
        #space['min_child_samples'] = int(space['min_child_samples'])

        #model = lgb.train(space, self.dtrain, space['num_rounds'], valid_sets = [self.dtest], verbose_eval = True)
        regr = lgb.LGBMRegressor(**space, n_jobs = 24, verbose = 0)
        regr.fit(self.X_train, self.y_train, eval_set=evaluation)
        

        pred = regr.predict(self.X_test)
        mse = mean_squared_error(self.y_test, pred)
        #print ("MSE:", mse)
        return {'loss': mse, 'status': STATUS_OK }


from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class LinearRegressor(LinearRegression):
    def __init__(self, target = 'LinearRegressor', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def fit(self, X, y):
        return super().fit(X, y)
    
    def predict(self, X):
        return super().predict(X)
    
    def plot(self, X, y, elevation_for_scaling = None, figsize=(8, 4)):
        y_pred = self.predict(X)
        residuals = (y - y_pred)

        if elevation_for_scaling is not None:
            residuals *= np.cos(elevation_for_scaling)
        #print mean residuals in degrees
        print(f'RMS {self.target}: {np.sqrt( np.mean( np.rad2deg(residuals* 3600 )**2 ) ):.2f} arcsecs')
        
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        ax1, ax2 = axes
        
        ax1.scatter(y, y_pred)
        ax1.set(xlabel='True Values', ylabel='Predictions', title='True vs. Predicted Values')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        
        ax2.hist(residuals, bins=25)
        ax2.set(xlabel='Residuals', ylabel='Frequency', title='Residual Histogram')
        
        plt.tight_layout()
        plt.savefig(f'./FinalResultsOptical/Plots/{self.target}_performance.png',dpi = 400)

    def plot_residuals(self, X, y):
        #Make plot with residuals and X[:,0] on the x-axis
        y_pred = self.predict(X)
        residuals = y - y_pred
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], residuals)
        ax.set(xlabel='X', ylabel='Residuals', title='Residuals vs. X')
        plt.tight_layout()
        plt.savefig(f'./FinalResultsOptical/Plots/{self.target}_residuals.png',dpi = 400)
        return


import torch 
def pytorch_model_wrapper(model, input_data):

    input_tensor = torch.from_numpy(input_data).float()
    output_tensor = model(input_tensor)
    output = output_tensor.detach().numpy()
    return output






def prepare_linear_terms(Az,El):
    terms_az = [
        Az,
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
        np.ones(len(Az)),
    ]
    terms_el = [
        El,
        np.sin(El),
        np.cos(El),
        np.cos(2*Az),
        np.sin(2*Az),
        np.cos(3*Az),
        np.sin(3*Az),
        np.sin(4*Az),
        np.sin(5*Az),
        np.sin(Az),
        np.sin(Az)*np.tan(El),
        np.ones(len(Az)),
    ]

    #Set of all terms in terma_az and terms_el
    terms_both = [
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
        np.ones(len(Az)),
        np.sin(El),
        np.cos(El),
        np.sin(3*Az),
        np.sin(4*Az),
        np.sin(5*Az),
        np.cos(2*Az) / np.cos(El),
        np.cos(Az) / np.cos(El),
        np.cos(5*Az) / np.cos(El)
    ]

    terms_az = np.column_stack(terms_az)
    terms_el = np.column_stack(terms_el)
    terms_both = np.column_stack(terms_both)
    
    return terms_az, terms_el, terms_both


def correlation_plot(df):
    """
    Plot the correlation of residuals with the features using seaborn.
    Residuals are 1d numpy array and features are dataframe.
    """
    state = 'ALL'
    df.loc[df.SUNEL_MEDIAN_1 < 0. , 'SUNEL_MEDIAN_1'] = 0.
    df.loc[df.SUNEL_MEDIAN_1 < 0. , 'SUNAZ_MEDIAN_1'] = 0.
    df['HOUR'] = df['date'].dt.hour
    df['MINUTE'] = df['date'].dt.minute
    # df = df[df.SUNEL_MEDIAN_1 > 0]
    print(f'NUMBER OF DATAPOINTS DURING {state} = {len(df)}')
    print(df.describe())
    corr = df.corr(method = 'pearson')
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(25, 18))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, annot_kws={"size": 14})
    plt.savefig(f'./AnalyticalModelRaw/Plots/Correlation_residuals_{state.lower()}.png', dpi = 400)

from dataset import PrepareDataRaw, PrepareDataForNewModel, PrepareDataRaw_v2, PrepareDataFinal

def fit_linear_models(tp_key = 0):

    # Read the data
    # df_features = pd.read_csv('./Data/processed_optical/features_optical.csv')
    # df_features['date'] = pd.to_datetime(df_features['date'])
    # df_features = df_features[df_features.date < pd.Timestamp('2022-09-17')]
    # df_features = df_features[feats]

    # df = pd.read_csv('./Data/raw_nflash230.csv')
    # df['date'] = pd.to_datetime(df['date'])
    # df = df[df.date < pd.Timestamp('2022-09-17')]

    # df = df.merge(df_features, on='date', how='inner')
    # df = df.dropna().drop_duplicates()

    # Split the data into training and testing sets
    # ds = PrepareDataRaw(path_features = None, target_key = 'both', use_cartesian=False, use_scaler = False)
    path_df = './Data/raw_nflash230.csv'
    dataset_params['target'] = 'optical_both'
    time_period_tests = {
        0: (pd.Timestamp('2022-05-22 06:00:00'), pd.Timestamp('2022-05-22 23:40:00')),
        1: (pd.Timestamp('2022-05-22'), pd.Timestamp('2022-07-04')),
        2: (pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-09-17 17:30:00')),
        3: (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-05-22')),
        4: (pd.Timestamp('2022-07-05'), pd.Timestamp('2022-08-12')),
    }

    start, end = time_period_tests[tp_key]
    
    print(f'tp_key: {tp_key}, start: {start}, end: {end}')
    df = pd.read_csv(path_df)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start) & (df['date'] <= end)]

    df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']] = np.deg2rad(df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])

    X_train, X_test, y_train, y_test = train_test_split(df[['COMMANDAZ', 'COMMANDEL']], df[['ACTUALAZ', 'ACTUALEL']], test_size=0.2, random_state=random_seed)
    # ds = PrepareDataForNewModel(df_path = path_df, params = dataset_params)

    # COMMAND_train,COMMAND_test,y_train,y_test = ds.get_data()
    # # Fit the models to the training data

    # X_az_train, X_el_train, X_both_train = prepare_linear_terms(COMMAND_train['COMMANDAZ'], COMMAND_train['COMMANDEL'])
    # X_az_test, X_el_test, X_both_test = prepare_linear_terms(COMMAND_test['COMMANDAZ'], COMMAND_test['COMMANDEL'])

    X_az_train, X_el_train, X_both_train = prepare_linear_terms(X_train['COMMANDAZ'], X_train['COMMANDEL'])
    X_az_test, X_el_test, X_both_test = prepare_linear_terms(X_test['COMMANDAZ'], X_test['COMMANDEL'])
    
    y_az_train, y_el_train = y_train['ACTUALAZ'], y_train['ACTUALEL']
    y_az_test, y_el_test = y_test['ACTUALAZ'], y_test['ACTUALEL']

    model_az = LinearRegressor(target=f'az_tp{tp_key}')
    model_az.fit(X_az_train, y_train['ACTUALAZ'].values)
    model_az.plot(X_az_train, y_train['ACTUALAZ'].values, X_train['COMMANDEL'])
    model_az.plot_residuals(X_az_train, y_train['ACTUALAZ'].values)

    model_el = LinearRegressor(target='el_datasplit')
    model_el.fit(X_el_train, y_train['ACTUALEL'].values)
    model_el.plot(X_el_train, y_train['ACTUALEL'].values)
    model_el.plot_residuals(X_el_train, y_train['ACTUALEL'].values)

    # Predict on the testing set
    az_pred_test = model_az.predict(X_az_test)
    el_pred_test = model_el.predict(X_el_test)

    az_residuals_test = az_pred_test - y_test['ACTUALAZ'].values
    el_residuals_test = el_pred_test - y_test['ACTUALEL'].values


    az_residuals_test *= np.cos(X_test['COMMANDEL'].values)
    az_residuals_test = np.rad2deg(az_residuals_test) * 3600
    el_residuals_test = np.rad2deg(el_residuals_test) * 3600

    mean_error_test = np.sqrt(np.mean(az_residuals_test**2 + el_residuals_test**2))
    
    print(f'RMS on testing set: {mean_error_test:.2f} arcsecs')

    # Predict on the training set
    az_pred_train = model_az.predict(X_az_train)
    el_pred_train = model_el.predict(X_el_train)

    az_residuals_train = az_pred_train - y_train['ACTUALAZ'].values
    el_residuals_train = el_pred_train - y_train['ACTUALEL'].values

    az_residuals_train *= np.cos(X_train['COMMANDEL'].values)
    az_residuals_train = np.rad2deg(az_residuals_train) * 3600
    el_residuals_train = np.rad2deg(el_residuals_train) * 3600
    mean_error_train = np.sqrt(np.mean(az_residuals_train**2 + el_residuals_train**2))
    
    print(f'RMS on training set: {mean_error_train:.2f} arcsecs')

def linear_model_dump():
    # terms_az.append(df.date.dt.hour)
    # terms_az.append(df.TEMP26_MEDIAN_1)
    # terms_az.append(df.SUNAZ_MEDIAN_1)
    # terms_az.append(df.PRESSURE_MEDIAN_1)
    # terms_az.append(df.DISP_ABS1_MEDIAN_1)
    # terms_az.append(df.DISP_ABS3_MEDIAN_1)
    # terms_az.append(df.TILT1X_MEDIAN_1)
    # terms_az.append(df.POSITIONZ_MEDIAN_1)

    # terms_el.append(df.date.dt.hour)
    # terms_el.append(df.TEMP26_MEDIAN_1)
    # terms_el.append(df.SUNAZ_MEDIAN_1)
    # terms_el.append(df.DISP_ABS3_MEDIAN_1)
    
    df_features = pd.read_csv('./Data/processed_optical/features_optical.csv')
    df_features['date'] = pd.to_datetime(df_features['date'])
    df_features = df_features[df_features.date < pd.Timestamp('2022-09-17')]
    df_features = df_features[feats]
    
    
    df = pd.read_csv('./Data/raw_nflash230.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.date < pd.Timestamp('2022-09-17')]

    df = df.merge(df_features, on = 'date', how = 'inner')
    df = df.dropna().drop_duplicates()

    date = df.date
    df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']] = np.deg2rad(df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])
    print(len(df))
    terms_az, terms_el, terms_both = prepare_linear_terms(df['COMMANDAZ'], df['COMMANDEL'])
    print('Type dt.hour', type(df.date.dt.hour))



    model_az = LinearRegressor(target = 'az_v2')
    model_az.add_terms(terms_az)
    model_az.fit(df['ACTUALAZ'].values)
    model_az.plot(model_az.X, df['ACTUALAZ'].values, df.COMMANDEL.values)
    model_az.plot_residuals(model_az.X, df['ACTUALAZ'].values)
    model_el = LinearRegressor(target = 'el_v2')
    model_el.add_terms(terms_el)
    model_el.fit(df['ACTUALEL'].values)
    model_el.plot(model_el.X, df['ACTUALEL'].values)
    model_el.plot_residuals(model_el.X, df['ACTUALEL'].values)

    az_pred = model_az.predict(model_az.X)
    el_pred = model_el.predict(model_el.X)
    
    az_residuals = az_pred - df['ACTUALAZ'].values
    el_residuals = el_pred - df['ACTUALEL'].values

    az_residuals *= np.cos(df['COMMANDEL'].values) 

    mean_error = np.mean(np.sqrt(az_residuals**2 + el_residuals**2))
    mean_error = np.rad2deg(mean_error) * 3600
    print(f'Mean error: {mean_error:.2f} arcsecs')

    print("Fitted parameters for azimuth:")
    print(model_az.intercept_*3600)
    print(model_az.coef_*3600)
    print("Fitted parameters for elevation:")
    print(model_el.intercept_*3600)
    print(model_el.coef_*3600)

    df_residuals = pd.DataFrame({'date': date, 'RESIDUALSAZ': az_residuals, 'RESIDUALEL': el_residuals})
    df_residuals['date'] = pd.to_datetime(df_residuals['date'])



    #Merge df_residuals with df_features on date
    df_features = pd.merge(df_features, df_residuals, on = 'date', how = 'inner').dropna().drop_duplicates()
    correlation_plot(df_features)


def plot_sorted_predictions_models(model, X, y_true, y_scaler, name, azimuth = False):
    print(f"Plotting sorted predictions for {name}")

    PATH_SORTPRED = './PretrainedModel/SortedPrediction/'
    FULL_PATH_SORTPRED = os.path.join(PATH_SORTPRED, f'sortpred_{name}.png')
    
    
    y_pred = model.predict(X)

    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(y_pred)

    plt.clf()       

    residuals = y_true - y_pred
    residuals = np.rad2deg(residuals) * 3600
    
    if azimuth:
        residuals *= np.cos(X[:, 1])

    RMS = np.sqrt(np.mean( residuals**2 ))
    
    idxSorted = y_true[:,0].argsort()
    plt.plot(range(len(y_pred)), y_pred[idxSorted], label="Prediction")
    plt.plot(range(len(y_pred)), y_true[idxSorted], label='True')

    plt.xlabel("Sample #")
    plt.ylabel("Offset [arcseconds]")
    print(f'RMS for {name}: {RMS:.3f} arcsecs')
    plt.title(f"{name} | RMS: {RMS:.3f} arcsecs ")
    plt.legend()
    
    plt.savefig(FULL_PATH_SORTPRED, dpi = 400)

    return RMS

from settings import dataset_params


def XGB_Optical():

    path_df = './Data/raw_nflash230.csv'
    dataset_params['target'] = 'optical_az'
    ds_az = PrepareDataForNewModel(df_path = path_df, params = dataset_params)
    model_az = XGBoostRegressor(ds_az, name = 'XGB_optical_az', train_model = True, load_model = True)

    dataset_params['target'] = 'optical_el'
    ds_el = PrepareDataForNewModel(df_path = path_df, params = dataset_params)
    model_el = XGBoostRegressor(ds_az, name = 'XGB_optical_el', train_model = True, load_model = True)

    model_az.plot_sorted_prediction_v2(elevation_for_scaling = ds_az.X_test.iloc[:,1].values)
    model_el.plot_sorted_prediction_v2()
    
    model_az.plot_error_locations()
    model_el.plot_error_locations()  


def new_model_on_pointing_scans():
    """
    Train xgb model with the dataset that has the outputs from optical model.
    """

    dataset_params['target'] = 'residual_az'
    path_data = './Data/scans_nflash230_model_all_ts01.csv'
    ds_az = PrepareDataForNewModel(df_path = path_data, params = dataset_params)
    model_az = XGBoostRegressor(ds_az, name = 'XGB_az_new_all_ts01', train_model = True, load_model = True)

    dataset_params['target'] = 'residual_el'
    path_data = './Data/scans_nflash230_model_all_ts01.csv'
    ds_el = PrepareDataForNewModel(df_path = path_data, params = dataset_params)
    model_el = XGBoostRegressor(ds_el, name = 'XGB_el_new_all_ts01', train_model = True, load_model = True)

    model_az.plot_sorted_prediction_v2()#elevation_for_scaling = ds_az.X_test.iloc[:,1].values)
    model_el.plot_sorted_prediction_v2()
    
    model_az.plot_error_locations()
    model_el.plot_error_locations()

    # preds_az = model_az.predict(ds_az.X_test)
    # preds_el = model_el.predict(ds_el.X_test)

def test_xgb_on_pointing_scans():

    PATH_MODEL = f'./AnalyticalModelRaw/Models/'

    df = pd.read_csv('./Data/scans_nflash230_unscaled.csv')
    X = df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']].values
    y_az = df[['REALAZ']].values
    y_el = df[['REALEL']].values

    model_az = pickle.load(open(os.path.join(PATH_MODEL, 'XGB_optical_az.pkl'), 'rb'))
    model_el = pickle.load(open(os.path.join(PATH_MODEL, 'XGB_optical_el.pkl'), 'rb'))
    
    plot_sorted_predictions_models(model_az, X, y_az, y_scaler = None, name = 'XGB_optical_az', azimuth = True)
    plot_sorted_predictions_models(model_el, X, y_el, y_scaler = None, name = 'XGB_optical_el', azimuth = False)


def test_linreg_on_pointing_scans():
    path_df = './Data/raw_nflash230.csv'
    dataset_params['target'] = 'optical_both'
    ds = PrepareDataForNewModel(df_path = path_df, params = dataset_params)

    COMMAND_train,COMMAND_test,y_train,y_test = ds.get_data()
    # Fit the models to the training data

    X_az_train, X_el_train, X_both_train = prepare_linear_terms(COMMAND_train['COMMANDAZ'], COMMAND_train['COMMANDEL'])
    X_az_test, X_el_test, X_both_test = prepare_linear_terms(COMMAND_test['COMMANDAZ'], COMMAND_test['COMMANDEL'])

    
    model_az = LinearRegressor(target='az_datasplit')
    model_az.fit(X_az_train, y_train['ACTUALAZ'].values)

    model_el = LinearRegressor(target='el_datasplit')
    model_el.fit(X_el_train, y_train['ACTUALEL'].values)


    df = pd.read_csv('./Data/scans_nflash230_unscaled.csv')
    X = df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']].values
    y_az = df[['REALAZ']].values
    y_el = df[['REALEL']].values

    X_az_test, X_el_test, X_both_test = prepare_linear_terms(df['COMMANDAZ_MEDIAN'],df['COMMANDEL_MEDIAN'])

    plot_sorted_predictions_models(model_az, X_az_test, y_az, y_scaler = None, name = 'Linreg_optical_az', azimuth = True)
    plot_sorted_predictions_models(model_el, X_el_test, y_el, y_scaler = None, name = 'Linreg_optical_el', azimuth = False)

def add_xgb_model_output_to_dataset(path_dataset):
    PATH_MODEL = f'./AnalyticalModelRaw/Models/'

    df = pd.read_csv(path_dataset)
    X = df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']].values
    y_az = df[['REALAZ']].values
    y_el = df[['REALEL']].values

    model_az = pickle.load(open(os.path.join(PATH_MODEL, 'XGB_optical_az.pkl'), 'rb'))
    model_el = pickle.load(open(os.path.join(PATH_MODEL, 'XGB_optical_el.pkl'), 'rb'))
    
    preds_az = model_az.predict(X)
    preds_el = model_el.predict(X)

    df['MODELAZ'] = preds_az
    df['MODELEL'] = preds_el

    df['RESIDUALAZ'] = df['REALAZ'] - df['MODELAZ']
    df['RESIDUALEL'] = df['REALEL'] - df['MODELEL']


    df.to_csv(f'./Data/scans_nflash230_xgb_all.csv', index=False)

def add_linreg_output_to_dataset(path_dataset):
    path_df = './Data/raw_nflash230.csv'
    dataset_params['target'] = 'optical_both'
    ds = PrepareDataForNewModel(df_path = path_df, params = dataset_params)

    COMMAND_train,COMMAND_test,y_train,y_test = ds.get_data()
    # Fit the models to the training data

    X_az_train, X_el_train, X_both_train = prepare_linear_terms(COMMAND_train['COMMANDAZ'], COMMAND_train['COMMANDEL'])
    X_az_test, X_el_test, X_both_test = prepare_linear_terms(COMMAND_test['COMMANDAZ'], COMMAND_test['COMMANDEL'])

    
    model_az = LinearRegressor(target='az_datasplit')
    model_az.fit(X_az_train, y_train['ACTUALAZ'].values)

    model_el = LinearRegressor(target='el_datasplit')
    model_el.fit(X_el_train, y_train['ACTUALEL'].values)


    PATH_MODEL = f'./AnalyticalModelRaw/Models/'

    df = pd.read_csv(path_dataset)
    y_az = df[['REALAZ']].values
    y_el = df[['REALEL']].values

    X_az, X_el, X_both = prepare_linear_terms(df['COMMANDAZ_MEDIAN'], df['COMMANDEL_MEDIAN'])

    preds_az = model_az.predict(X_az)
    preds_el = model_el.predict(X_el)

    df['MODELAZ'] = preds_az
    df['MODELEL'] = preds_el

    df['RESIDUALAZ'] = df['REALAZ'] - df['MODELAZ']
    df['RESIDUALEL'] = df['REALEL'] - df['MODELEL']

    df.to_csv(f'./Data/scans_nflash230_linreg_all.csv', index=False)


def xgb_linear_terms():

    ds_az = PrepareDataRaw_v2(target_key = 'az')
    model_az = XGBoostRegressor(ds_az, name = 'XGB_optical_linterm_az', train_model = True, load_model = True)


    ds_el = PrepareDataRaw_v2(target_key = 'el')
    model_el = XGBoostRegressor(ds_az, name = 'XGB_optical_linterm_el', train_model = True, load_model = True)

    model_az.plot_sorted_prediction_v2(elevation_for_scaling = ds_az.X_test[:,1])
    model_el.plot_sorted_prediction_v2()
    
    model_az.plot_error_locations()
    model_el.plot_error_locations()  





"""

ACTUALAZ_MEDIAN,ACTUALEL_MEDIAN,WINDDIRECTION_MEDIAN,WINDSPEED_MEDIAN,DAZ_DISP_MEDIAN,DAZ_SPEM_MEDIAN,
DAZ_TEMP_MEDIAN,DAZ_TILT_MEDIAN,DAZ_TILTTEMP_MEDIAN,COMMANDAZ_MEDIAN,COMMANDEL_MEDIAN,DEL_SPEM_MEDIAN,
DEL_DISP_MEDIAN,DEL_TEMP_MEDIAN,DEL_TILT_MEDIAN,DEL_TILTTEMP_MEDIAN,SUNAZ_MEDIAN,SUNEL_MEDIAN,TILT1T_MEDIAN,
TILT1X_MEDIAN,TILT1Y_MEDIAN,TILT2T_MEDIAN,TILT2X_MEDIAN,TILT2Y_MEDIAN,TILT3T_MEDIAN,TILT3X_MEDIAN,TILT3Y_MEDIAN,
TEMP1_MEDIAN,TEMP2_MEDIAN,TEMP3_MEDIAN,TEMP4_MEDIAN,TEMP5_MEDIAN,TEMP6_MEDIAN,TEMP26_MEDIAN,TEMP27_MEDIAN,TEMP28_MEDIAN,
TEMPERATURE_MEDIAN,POSITIONX_MEDIAN,POSITIONY_MEDIAN,POSITIONZ_MEDIAN,ROTATIONX_MEDIAN,ROTATIONY_MEDIAN,HUMIDITY_MEDIAN,
DEWPOINT_MEDIAN,PRESSURE_MEDIAN,DEL_TOTAL_MEDIAN,DAZ_TOTAL_MEDIAN,DISP_ABS1_MEDIAN,DISP_ABS2_MEDIAN,DISP_ABS3_MEDIAN,
WINDSPEED_VARIANCE_P5,DSUNAZ_CHANGE_P5,rx,ca,ie,Off_Az,Off_El,month_continuous,month,time_of_day
"""
"""
WINDDIRECTION_CHANGE_I1,TEMP1_CHANGE_I1,TEMP26_CHANGE_I1,TILT1T_CHANGE_I1,COMMANDAZ_CHANGE_I1,COMMANDEL_CHANGE_I1,
ACTUALAZ_CHANGE_I1,ACTUALEL_CHANGE_I1,DAZ_DISP_CHANGE_I1,DAZ_SPEM_CHANGE_I1,DAZ_TEMP_CHANGE_I1,DAZ_TILT_CHANGE_I1,
DAZ_TILTTEMP_CHANGE_I1,DEL_SPEM_CHANGE_I1,DEL_DISP_CHANGE_I1,DEL_TEMP_CHANGE_I1,DEL_TILT_CHANGE_I1,DEL_TILTTEMP_CHANGE_I1,
ACTUALAZ_POS_CHANGE,ACTUALEL_POS_CHANGE,COMMANDAZ_POS_CHANGE,COMMANDEL_POS_CHANGE,ACTUALAZ_NEG_CHANGE,ACTUALEL_NEG_CHANGE,
COMMANDAZ_NEG_CHANGE,COMMANDEL_NEG_CHANGE,ACTUALAZ_MEDIAN,ACTUALEL_MEDIAN,WINDDIRECTION_MEDIAN,WINDSPEED_MEDIAN,DAZ_DISP_MEDIAN,
DAZ_SPEM_MEDIAN,DAZ_TEMP_MEDIAN,DAZ_TILT_MEDIAN,DAZ_TILTTEMP_MEDIAN,COMMANDAZ_MEDIAN,COMMANDEL_MEDIAN,DEL_SPEM_MEDIAN,DEL_DISP_MEDIAN,
DEL_TEMP_MEDIAN,DEL_TILT_MEDIAN,DEL_TILTTEMP_MEDIAN,SUNAZ_MEDIAN,SUNEL_MEDIAN,TILT1T_MEDIAN,TILT1X_MEDIAN,TILT1Y_MEDIAN,TILT2T_MEDIAN,
TILT2X_MEDIAN,TILT2Y_MEDIAN,TILT3T_MEDIAN,TILT3X_MEDIAN,TILT3Y_MEDIAN,TEMP1_MEDIAN,TEMP2_MEDIAN,TEMP3_MEDIAN,TEMP4_MEDIAN,TEMP5_MEDIAN,
TEMP6_MEDIAN,TEMP26_MEDIAN,TEMP27_MEDIAN,TEMP28_MEDIAN,TEMPERATURE_MEDIAN,POSITIONX_MEDIAN,POSITIONY_MEDIAN,POSITIONZ_MEDIAN,
ROTATIONX_MEDIAN,ROTATIONY_MEDIAN,HUMIDITY_MEDIAN,DEWPOINT_MEDIAN,PRESSURE_MEDIAN,DEL_TOTAL_MEDIAN,DAZ_TOTAL_MEDIAN,DISP_ABS1_MEDIAN,
DISP_ABS2_MEDIAN,DISP_ABS3_MEDIAN,WINDSPEED_VARIANCE_P5,TEMP1_MAX_CHANGE_I5,TEMP26_MAX_CHANGE_I5,ACTUALVELOCITYAZ_MAX,
ACTUALVELOCITYEL_MAX,DSUNAZ_CHANGE_P5,rx,ca,ie,Off_Az,Off_El,hours_since_corr,month_continuous,month,time_of_day
"""
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def get_selected_features(df_tmp, k, target_key, remove_correlated, path_results, name):
    # Feature selection using mutual information
    target = 'Off_Az' if target_key == 'az' else 'Off_El'
    other_target = 'Off_El' if target_key == 'az' else 'Off_Az'
    X = df_tmp.loc[ : , ~df_tmp.columns.isin([target, other_target, 'date', 'day', 'rx'])]
    y = df_tmp[target]
    selector = SelectKBest(mutual_info_regression, k=k)
    try:
        selector.fit(X, y)
    except:
        embed()
    selected_features = X.columns[selector.get_support()].to_list()

    if remove_correlated:
        # Removal of highly correlated features
        selected_features_not_correlated = []
        corr_matrix = np.abs(df_tmp[selected_features].corr())
        for i in range(len(selected_features)):
            if selected_features[i] not in selected_features_not_correlated:
                corr_values = corr_matrix[selected_features[i]][selected_features].values
                idx = np.argwhere(corr_values > 0.75)
                if len(idx) > 1:
                    mean_corr = np.mean(corr_values[idx])
                    for j in idx[:, 0]:
                        if corr_values[j] >= mean_corr:

                            selected_features_not_correlated.append(selected_features[j])
                else:
                    selected_features_not_correlated.append(selected_features[i])
        selected_features = selected_features_not_correlated
    
    df_feature_list = pd.DataFrame(data={'features':selected_features})
    df_feature_list.to_csv(os.path.join(path_results, f'SelectedFeatures_v2/{name}'), index=False)
    return selected_features

def XGB_experiment(process_number = 99):
    
    run_number = 1

    path_result = f'./FinalResults/Run{run_number}/' 
    if not os.path.exists(path_result):
        os.makedirs(path_result)


    time_period_tests = {
        1: (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-09-17')),
        2: (pd.Timestamp('2022-05-23'), pd.Timestamp('2022-07-04')),
        3: (pd.Timestamp('2022-07-06'), pd.Timestamp('2022-08-18')),
        4: (pd.Timestamp('2022-08-20'), pd.Timestamp('2022-09-17')),
    }
    
    datasets = [fn for fn in os.listdir('./Datasets/') if not fn.endswith('.csv')]
    dataset_tests = {i: f'./Datasets/{fn}/' for i, fn in enumerate(datasets)}
    # create an empty DataFrame
    df_model_params = pd.DataFrame(columns=['dataset_key', 'timeperiod_key','target','num_features', 'Model Name', 'Model Parameters', 'RMS Model', 'RMS Current', 'RMS Compared'])

    key_timeperiod = int(process_number)
    timeperiod_tuple = time_period_tests[key_timeperiod]
    count = 1
    for key_dataset, dataset_path in dataset_tests.items():
        for target_key in ['az', 'el']:
            for k in [5,10,15,20,25,30,40,50]:
                for remove_correlated in [False]:#, True]:
                    
                    print(f'key ds: {dataset_path}, target: {target_key}, k: {k}, remove corr: {remove_correlated}, count: {count}')
                    count += 1

    print(count)
    return
    for key_dataset, dataset_path in dataset_tests.items():
        print(f'Dataset {key_dataset} | {datasets[key_dataset]}...')
        df = pd.read_csv(os.path.join(dataset_path, 'features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])

        print(f'Time Period {key_timeperiod} | {timeperiod_tuple}...')
    
        df_tmp = df[(df['date'] >= timeperiod_tuple[0]) & (df['date'] <= timeperiod_tuple[1])]
        df_tmp.dropna(axis=1, thresh=len(df_tmp)-67, inplace=True)
        df_tmp.dropna(inplace=True)


        for target_key in ['az', 'el']:
            for k in [5,10,15,20,25,30,40,50]:
                for remove_correlated in [False, True]:
                    feature_list = get_selected_features(df_tmp, k, target_key, remove_correlated, dataset_path, key_timeperiod)
                    # df_feature_list = pd.read_csv(os.path.join(dataset_path, f'feature_list_tp{key_timeperiod}_{target_key}_k{k}_{"uncorr" if remove_correlated else "corr"}.csv'))
                    # feature_list = df_feature_list['features'].to_list()
                    #Add 'rx' to feature list unless there is only one unique rx
                    if len(df_tmp['rx'].unique()) > 1:
                        feature_list.append('rx')

                    ds = PrepareDataFinal(
                        df = df_tmp,
                        parameter_keys = (key_dataset, key_timeperiod),
                        feature_list = feature_list,
                        target_key = target_key,
                        run_number = run_number,
                    )

                    model = XGBoostRegressor(ds, name = f'XGB_ds{key_dataset}_tp{key_timeperiod}_k{k}_{"uncorr" if remove_correlated else "corr"}_{target_key}')

                    model_parameters = model.train()
                    RMS_model, RMS_current = model.plot_sorted_prediction_final()

                    model.plot_histogram()

                    RMS_compared = RMS_model / RMS_current
                    # add a new row to the DataFrame with the model parameters, keys, and performance measures
                    df_model_params = df_model_params.append({
                        'dataset_key': key_dataset,
                        'timeperiod_key': key_timeperiod,
                        'target': target_key,
                        'num_features': k,
                        'Model Name': model.name,
                        'Model Parameters': model_parameters,
                        'RMS Model': RMS_model,
                        'RMS Current': RMS_current,
                        'RMS Compared': RMS_compared,
                    }, ignore_index=True)
                    
                    # save the DataFrame to a CSV file
                    df_model_params.to_csv(path_result + f'model_parameters_{process_number}.csv', index=False)



def print_final_results_from_offset_prediction():
    datasets = [fn for fn in os.listdir('./Datasets/') if not fn.endswith('.csv')]
    dataset_tests = {i: f'./Datasets/{fn}/' for i, fn in enumerate(datasets)}
    
    df = pd.read_csv('./FinalResults/Run1/model_parameters_all.csv')
    time_period_tests = {
        1: (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-09-17')),
        2: (pd.Timestamp('2022-05-23'), pd.Timestamp('2022-07-04')),
        3: (pd.Timestamp('2022-07-06'), pd.Timestamp('2022-08-18')),
        4: (pd.Timestamp('2022-08-20'), pd.Timestamp('2022-09-17')),
    }
    for key_dataset, dataset_path in dataset_tests.items():
        for key_timeperiod in time_period_tests:
            for target_key in ['az', 'el']:

                df_tmp = df[(df['dataset_key'] == key_dataset) & (df['timeperiod_key'] == key_timeperiod) & (df['target'] == target_key)].copy()
                df_tmp.sort_values(by='RMS Compared', inplace=True)
                print(df_tmp.head(10))
                

    embed()
"""
CV short
6
tmp2022_clean_clf_transformed
tmp2022_clean_clf_nflash230_transformed
tmp2022_clean_clf
tmp2022_clean_clf_nflash230
7
tmp2022_clean_clf_transformed split days
tmp2022_clean_clf_nflash230_transformed split days
tmp2022_clean_clf split days
tmp2022_clean_clf_nflash230 split days
CV all 
8
tmp2022_clean_clf_transformed
tmp2022_clean_clf_nflash230_transformed
tmp2022_clean_clf
tmp2022_clean_clf_nflash230
9
tmp2022_clean_clf_transformed split days
tmp2022_clean_clf_nflash230_transformed split days
tmp2022_clean_clf split days
tmp2022_clean_clf_nflash230 split days

"""

def XGB_experiment_CV_short(process_number = 99):
    
    run_number = 11

    path_results = f'./FinalResults/Run{run_number}/' 
    for _path in [path_results, path_results + 'SelectedFeatures/']:
        if not os.path.exists(_path):
            os.makedirs(_path)


    
    path_datasets = './Datasets/'
    datasets = [(2,'tmp2022_clean_clf_nflash230'),(4, 'tmp2022_clean_clf_transformed'), (6, 'tmp2022_clean_clf_nflash230_transformed'), (1,'tmp2022_clean_clf')]


    # create an empty DataFrame
    df_model_params = pd.DataFrame(columns=['dataset', 'fold','target','num_features', 'Model Name', 'Model Parameters', 'Val RMS Model', 'Val RMS Current', 'Val RMS Compared', 'Test RMS Model', 'Test RMS Current', 'Test RMS Compared'])

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']
    for dataset_key,dataset in datasets:
        df = pd.read_csv(os.path.join(path_datasets, f'{dataset}/features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(axis=1, thresh=len(df)-67, inplace=True)
        df.dropna(inplace=True)

        n_folds = 6
        test_size = 1/n_folds
        df_folds = np.array_split(df, n_folds)

        for i,df_tmp in enumerate(df_folds):
            #Split into 0.7 0.2 0.1.
            n = len(df_tmp)
            df_trainval = df.iloc[:int((1-test_size)*n)]
            df_test = df.iloc[int((1-test_size)*n):]

            # embed(header = 'first loop xgb cv')
            for target_key in ['az', 'el']:
                for k in [2,5,10,20,30,40,50]:      
                    for remove_correlated in [True]:
                        model_name = f'XGB_ds{dataset_key}_tp{i}_k{k}_{"uncorr" if remove_correlated else "corr"}_{target_key}'

                        feature_list = get_selected_features(df_trainval[~df_trainval.isin(constant_features)].copy(), k, target_key, remove_correlated, path_results, model_name)
                        feature_list += constant_features
                        feature_list = list(dict.fromkeys(feature_list))
                        if len(df_tmp['rx'].unique()) > 1:
                            feature_list.append('rx')

                        ds = PrepareDataFinal(
                            df = df_trainval.copy(),
                            parameter_keys = (process_number, dataset_key),
                            feature_list = feature_list,
                            target_key = target_key,
                            run_number = run_number,
                        )

                        model = XGBoostRegressor(ds, name = model_name)
                        model_parameters = model.train()
                        model.plot_histogram()
                        RMS_val_model, RMS_val_current = model.plot_sorted_prediction_final()
                        RMS_val_compared = RMS_val_model / RMS_val_current
                        
                        #Test model on df_test
                        X = df_test[feature_list]
                        if 'rx' in X.columns:
                            le = LabelEncoder()
                            X['rx'] = le.fit_transform(X['rx'])
                        X = X.values
                        y = df_test['Off_Az' if target_key == 'az' else 'Off_El'].values

                        y_pred = model.model.predict(X)

                        RMS_test_model = np.sqrt(mean_squared_error(y, y_pred))
                        RMS_test_current = np.sqrt(mean_squared_error(y, np.zeros_like(y)))
                        RMS_test_compared = RMS_test_model / RMS_test_current

            
                        # add a new row to the DataFrame with the model parameters, keys, and performance measures
                        df_model_params = df_model_params.append({
                            'dataset': dataset,
                            'fold': i,
                            'target': target_key,
                            'num_features': k,
                            'Model Name': model_name,
                            'Model Parameters': model_parameters,
                            'Val RMS Model': RMS_val_model,
                            'Val RMS Current': RMS_val_current,
                            'Val RMS Compared': RMS_val_compared,
                            'Test RMS Model': RMS_test_model,
                            'Test RMS Current': RMS_test_current,
                            'Test RMS Compared': RMS_test_compared,
                        }, ignore_index=True)
                        

                        # save the DataFrame to a CSV file
                        df_model_params.to_csv(path_results + f'model_parameters_{process_number}.csv', index=False)

def get_split_indices(l, n):
    """
    Returns a list of cumulative indices to split an array of length l into n sub-arrays.
    The sub-arrays contain approximately the same number of rows.
    """
    indices = [0]
    size = l // n
    remainder = l % n
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + size + 1
        else:
            end = start + size
        indices.append(end)
        start = end

    return indices

def XGB_experiment_CV_all(process_number = 99, run_number = 99, split_on_days = False):
    

    path_results = f'./FinalResults/Run{run_number}/' 
    for _path in [path_results, path_results + 'SelectedFeatures/']:
        if not os.path.exists(_path):
            os.makedirs(_path)


    
    path_datasets = './Datasets/'
    datasets = [(2,'tmp2022_clean_clf_nflash230'),(4, 'tmp2022_clean_clf_transformed'), (6, 'tmp2022_clean_clf_nflash230_transformed'), (1,'tmp2022_clean_clf')]

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']

    # create an empty DataFrame
    df_model_params = pd.DataFrame(columns=['dataset', 'fold','target','num_features', 'Model Name', 'Model Parameters', 'Val RMS Model', 'Val RMS Current', 'Val RMS Compared', 'Test RMS Model', 'Test RMS Current', 'Test RMS Compared'])

    n_folds = 6

    for dataset_key,dataset in datasets:
        df = pd.read_csv(os.path.join(path_datasets, f'{dataset}/features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(axis=1, thresh=len(df)-67, inplace=True)
        df.dropna(inplace=True)

        n = len(df)
        #List on split sizes, n%n_folds with length n//n_folds + 1 and n_folds - n%n_folds with length n//n_folds, but each on is also the sum of the list until that point
        split_indices = get_split_indices(n, n_folds)

        for i in range(n_folds):
            df_test = df.iloc[split_indices[i] : split_indices[i+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[i]], df.iloc[split_indices[i+1]:]])

            # embed(header = 'first loop xgb cv all')
            for target_key in ['az', 'el']:
                for k in [2,5,10,20,30,40,50]:      
                    for remove_correlated in [True]:
                        model_name = f'XGB_ds{dataset_key}_tp{i}_k{k}_{"uncorr" if remove_correlated else "corr"}_{target_key}'

                        feature_list = get_selected_features(df_trainval[~df_trainval.isin(constant_features)].copy(), k, target_key, remove_correlated, path_results, model_name)
                        feature_list += constant_features
                        feature_list = list(dict.fromkeys(feature_list))

                        if len(df_trainval['rx'].unique()) > 1:
                            feature_list.append('rx')

                        ds = PrepareDataFinal(
                            df = df_trainval.copy(),
                            parameter_keys = (process_number, dataset_key),
                            feature_list = feature_list,
                            target_key = target_key,
                            run_number = run_number,
                            split_on_days = split_on_days
                        )

                        model = XGBoostRegressor(ds, name = model_name)
                        model_parameters = model.train()
                        model.plot_histogram()
                        RMS_val_model, RMS_val_current = model.plot_sorted_prediction_final()
                        RMS_val_compared = RMS_val_model / RMS_val_current
                        
                        #Test model on df_test
                        X = df_test[feature_list]
                        if 'rx' in X.columns:
                            le = LabelEncoder()
                            X['rx'] = le.fit_transform(X['rx'])
                        X = X.values
                        y = df_test['Off_Az' if target_key == 'az' else 'Off_El'].values

                        y_pred = model.model.predict(X)

                        RMS_test_model = np.sqrt(mean_squared_error(y, y_pred))
                        RMS_test_current = np.sqrt(mean_squared_error(y, np.zeros_like(y)))
                        RMS_test_compared = RMS_test_model / RMS_test_current

            
                        # add a new row to the DataFrame with the model parameters, keys, and performance measures
                        df_model_params = df_model_params.append({
                            'dataset': dataset,
                            'fold': i,
                            'target': target_key,
                            'num_features': k,
                            'Model Name': model_name,
                            'Model Parameters': model_parameters,
                            'Val RMS Model': RMS_val_model,
                            'Val RMS Current': RMS_val_current,
                            'Val RMS Compared': RMS_val_compared,
                            'Test RMS Model': RMS_test_model,
                            'Test RMS Current': RMS_test_current,
                            'Test RMS Compared': RMS_test_compared,
                        }, ignore_index=True)
                    

                    # save the DataFrame to a CSV file
                    # df_model_params.to_csv(path_results + f'model_parameters_{process_number}.csv', index=False)


def test_set_best_folds():
    path_df = './FinalResults/Run2/model_parameters_1.csv'

    df = pd.read_csv(path_df)

    #For each of the 6 folds, get the best model for each target
    for i in range(6):
        df_fold = df[df['fold'] == i]
        df_fold_target_az = df_fold[df_fold['target'] == 'az']
        df_fold_target_el = df_fold[df_fold['target'] == 'el']

        #get the name of the best model
        best_model_name_az = df_fold_target_az[df_fold_target_az['RMS Compared'] == df_fold_target_az['RMS Compared'].min()]['Model Name'].values[0]
        best_model_name_el = df_fold_target_el[df_fold_target_el['RMS Compared'] == df_fold_target_el['RMS Compared'].min()]['Model Name'].values[0]

        #Load models from pickle, using ./FinalResults/Run2/Models/ + name
        model_az = pickle.load(open(f'./FinalResults/Run2/Models/{best_model_name_az}.pkl', 'rb'))
        model_el = pickle.load(open(f'./FinalResults/Run2/Models/{best_model_name_el}.pkl', 'rb'))

        y_az_pred = model_az.predict(X)



def print_final_results_from_offset_prediction():
    datasets = [fn for fn in os.listdir('./Datasets/') if not fn.endswith('.csv')]
    dataset_tests = {i: f'./Datasets/{fn}/' for i, fn in enumerate(datasets)}
    
    df = pd.read_csv('./FinalResults/Run1/model_parameters_all.csv')
    time_period_tests = {
        1: (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-09-17')),
        2: (pd.Timestamp('2022-05-23'), pd.Timestamp('2022-07-04')),
        3: (pd.Timestamp('2022-07-06'), pd.Timestamp('2022-08-18')),
        4: (pd.Timestamp('2022-08-20'), pd.Timestamp('2022-09-17')),
    }
    for key_dataset, dataset_path in dataset_tests.items():
        for key_timeperiod in time_period_tests:
            for target_key in ['az', 'el']:

                df_tmp = df[(df['dataset_key'] == key_dataset) & (df['timeperiod_key'] == key_timeperiod) & (df['target'] == target_key)].copy()
                df_tmp.sort_values(by='RMS Compared', inplace=True)
                print(df_tmp.head(10))
                

    embed()

def run_sage_for_some_models():
    """
    To be implemented. Should run SAGE for some models.
    Flexible funciton that makes it easy.
    """
    pass


def analyze_results_corrections():

    # df_short2 = pd.read_csv('./FinalResults/Run10/model_parameters_10.csv')
    # df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')
    # df_all2 = pd.read_csv('./FinalResults/Run10/model_parameters_12.csv')
    # df_all_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_13.csv')

    # df_short = pd.read_csv('./FinalResults/Run6/model_parameters_6.csv')
    # df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')
    # df_all = pd.read_csv('./FinalResults/Run6/model_parameters_8.csv')
    # df_all_days = pd.read_csv('./FinalResults/Run6/model_parameters_9.csv')

    # #Append df2's to the end of the originals
    # df_short = df_short.append(df_short2)
    # df_short_days = df_short_days.append(df_short_days2)
    # df_all = df_all.append(df_all2)
    # df_all_days = df_all_days.append(df_all_days2)


    df_all_days = pd.read_csv('./FinalResults/Run11/model_parameters_1.csv')
    df_all_random = pd.read_csv('./FinalResults/Run11/model_parameters_2.csv')
    df_all_days_actual = pd.read_csv('./FinalResults/Run11/model_parameters_3.csv')
    df_all_random_actual = pd.read_csv('./FinalResults/Run11/model_parameters_4.csv')



    dfs = [df_all_days, df_all_random, df_all_days_actual, df_all_random_actual]
    # descr = ['Short timeperiods, randomly split', 'Short timeperiods, split on days', 'Long time periods, randomly split', 'Long time periods, split on days']
    descr = ['Long period, split on days, command', 'Long period, randomly split, command', 'Long period, split on days, actual', 'Long period, randomly split, actual']
    for _descr, df in zip(descr, dfs):
        
        print(_descr)
        min_val_idx = df.groupby(['dataset', 'fold', 'target'])['Val RMS Compared'].idxmin()
        min_val = df.loc[min_val_idx]

        for ds in df.dataset.unique():
            for target in df.target.unique():
                df_tmp = min_val[(min_val['dataset'] == ds) & (min_val['target'] == target)]
                
                print(f'Dataset: {ds.ljust(39)}, Target: {target}, Mean compared test RMS: {df_tmp["Test RMS Compared"].mean():.3f}, STD: {df_tmp["Test RMS Compared"].std():.3f}')


def analyze_results_corrections_k():

    # df_short2 = pd.read_csv('./FinalResults/Run10/model_parameters_10.csv')
    # df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')
    # df_all2 = pd.read_csv('./FinalResults/Run10/model_parameters_12.csv')
    # df_all_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_13.csv')

    # df_short = pd.read_csv('./FinalResults/Run6/model_parameters_6.csv')
    # df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')
    # df_all = pd.read_csv('./FinalResults/Run6/model_parameters_8.csv')
    # df_all_days = pd.read_csv('./FinalResults/Run6/model_parameters_9.csv')


    # #Append df2's to the end of the originals
    # df_short = df_short.append(df_short2)
    # df_short_days = df_short_days.append(df_short_days2)
    # df_all = df_all.append(df_all2)
    # df_all_days = df_all_days.append(df_all_days2)

    # df_all_days = pd.read_csv('./FinalResults/Run11/model_parameters_1.csv')
    # df_all_random = pd.read_csv('./FinalResults/Run11/model_parameters_2.csv')
    # df_all_days_actual = pd.read_csv('./FinalResults/Run11/model_parameters_3.csv')
    # df_all_random_actual = pd.read_csv('./FinalResults/Run11/model_parameters_4.csv')

    df_all_days = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')
    df_all_random = pd.read_csv('./FinalResults/Run12/model_parameters_6.csv')
    df_all_days_actual = pd.read_csv('./FinalResults/Run12/model_parameters_7.csv')
    df_all_random_actual = pd.read_csv('./FinalResults/Run12/model_parameters_8.csv')

    dfs = [df_all_days, df_all_random, df_all_days_actual, df_all_random_actual]
    # descr = ['Short timeperiods, randomly split', 'Short timeperiods, split on days', 'Long time periods, randomly split', 'Long time periods, split on days']
    # descr = ['Long period, split on days, command', 'Long period, randomly split, command', 'Long period, split on days, actual', 'Long period, randomly split, actual']
    descr = ['days ts 0.4', 'random ts 0.4', 'days ts 0.5', 'random ts 0.5']
    for _descr, df in zip(descr, dfs):
        
        print(_descr)
        min_val_idx = df.groupby(['dataset', 'fold', 'target'])['Val RMS Compared'].idxmin()
        min_val = df.loc[min_val_idx]

        if not os.path.exists(_descr):
            os.makedirs(_descr)
        
        for ds in df.dataset.unique():
            k_list = []
            az_rms = []
            el_rms = []
            az_std = []
            el_std = []
            print(ds)
            print('k'.ljust(5), 'Target'.ljust(5), 'RMS'.ljust(5), 'STD'.ljust(5))
            for target in df.target.unique():
                df_k = df[(df['dataset'] == ds) & (df['target'] == target)]
                df_tmp = min_val[(min_val['dataset'] == ds) & (min_val['target'] == target)]
                #Mean val RMS compared and mean test RMS compared
                # print(f'Dataset: {ds.ljust(39)}, Target: {target}, Mean compared val RMS: {df_tmp["Val RMS Compared"].mean():.3f}, Mean compared test RMS: {df_tmp["Test RMS Compared"].mean():.3f}, STD: {df_tmp["Test RMS Compared"].std():.3f}')

                for k in df.num_features.unique():
                    df_k2 = df_k[df_k['num_features'] == k]
                    #Mean val RMS compared and mean test RMS compared
                    # print(f'k: {k}, Dataset: {ds.ljust(39)}, Target: {target}, Mean compared val RMS: {df_k2["Val RMS Compared"].mean():.3f}, Mean compared test RMS: {df_k2["Test RMS Compared"].mean():.3f}, STD: {df_k2["Test RMS Compared"].std():.3f}')
                    #Print k, target, mean compared test RMS and std, with not other text
                    print(f'{k}, {target}, {df_k2["Test RMS Compared"].mean():.3f}, {df_k2["Test RMS Compared"].std():.3f}')

                    if target == 'az':
                        az_rms.append(df_k2["Test RMS Compared"].mean())
                        az_std.append(df_k2["Test RMS Compared"].std())
                        k_list.append(k)
                    
                    if target == 'el':
                        el_rms.append(df_k2["Test RMS Compared"].mean())
                        el_std.append(df_k2["Test RMS Compared"].std())
            
            df_output = pd.DataFrame({'k': k_list, 'Az RMS': az_rms, 'Az STD': az_std, 'El RMS': el_rms, 'El STD': el_std})
            path_output = os.path.join(_descr, ds + '_results_table' '.csv')
            df_output.to_latex(path_output, index = False)
                
def analyze_results_cases():
    df_short2 = pd.read_csv('./FinalResults/Run10/model_parameters_10.csv')
    df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')
    # df_all2 = pd.read_csv('./FinalResults/Run10/model_parameters_12.csv')
    # df_all_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_13.csv')

    df_short = pd.read_csv('./FinalResults/Run6/model_parameters_6.csv')
    df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')
    # df_all = pd.read_csv('./FinalResults/Run6/model_parameters_8.csv')
    # df_all_days = pd.read_csv('./FinalResults/Run6/model_parameters_9.csv')

    # #Append df2's to the end of the originals
    df_short = df_short.append(df_short2)
    df_short_days = df_short_days.append(df_short_days2)
    # df_all = df_all.append(df_all2)
    # df_all_days = df_all_days.append(df_all_days2)


    # dfs = [df_short, df_short_days, df_all, df_all_days]
    # descr = ['Short timeperiods, randomly split', 'Short timeperiods, split on days', 'Long time periods, randomly split', 'Long time periods, split on days']
    




    df_all_days = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')
    df_all_random = pd.read_csv('./FinalResults/Run12/model_parameters_6.csv')
    df_all_days_actual = pd.read_csv('./FinalResults/Run12/model_parameters_7.csv')
    df_all_random_actual = pd.read_csv('./FinalResults/Run12/model_parameters_8.csv')

    dfs = [df_all_days, df_all_random, df_all_days_actual, df_all_random_actual]
    # descr = ['Short timeperiods, randomly split', 'Short timeperiods, split on days', 'Long time periods, randomly split', 'Long time periods, split on days']
    # descr = ['Long period, split on days, command', 'Long period, randomly split, command', 'Long period, split on days, actual', 'Long period, randomly split, actual']
    descr = ['PcorrResultsDays04', 'PcorrResultsRandom04', 'PcorrResultsDays05', 'PcorrResultsRandom05']

    # randomly_split = [df_short, df_all]
    # split_on_days = [df_short_days, df_all_days]
    randomly_split = [df_all_random, df_all_random_actual]
    split_on_days = [df_all_days, df_all_days_actual]

    descr_random = 'FinalTableRandomySplit'
    descr_days = 'FinalTableSplitOnDays'

    if not os.path.exists(descr_days):
        os.makedirs(descr_days)
    if not os.path.exists(descr_random):
        os.makedirs(descr_random)


    for ds in df_short.dataset.unique():
        print(ds)
        k_list = []
        az_rms1 = []
        az_rms2 = []
        el_rms1 = []
        el_rms2 = []
        az_std1 = []
        az_std2 = []
        el_std1 = []
        el_std2 = []
        for i, df in enumerate(split_on_days):


            for target in df.target.unique():
                df_k = df[(df['dataset'] == ds) & (df['target'] == target)]
                #Mean val RMS compared and mean test RMS compared
                # print(f'Dataset: {ds.ljust(39)}, Target: {target}, Mean compared val RMS: {df_tmp["Val RMS Compared"].mean():.3f}, Mean compared test RMS: {df_tmp["Test RMS Compared"].mean():.3f}, STD: {df_tmp["Test RMS Compared"].std():.3f}')

                for k in df.num_features.unique():
                    df_k2 = df_k[df_k['num_features'] == k]
                    #Mean val RMS compared and mean test RMS compared
                    # print(f'k: {k}, Dataset: {ds.ljust(39)}, Target: {target}, Mean compared val RMS: {df_k2["Val RMS Compared"].mean():.3f}, Mean compared test RMS: {df_k2["Test RMS Compared"].mean():.3f}, STD: {df_k2["Test RMS Compared"].std():.3f}')
                    #Print k, target, mean compared test RMS and std, with not other text

                    if target == 'az':
                        if i == 0:
                            az_rms1.append(df_k2["Test RMS Compared"].mean())
                            az_std1.append(df_k2["Test RMS Compared"].std())
                            k_list.append(k)
                        elif i == 1:
                            az_rms2.append(df_k2["Test RMS Compared"].mean())
                            az_std2.append(df_k2["Test RMS Compared"].std())
                    
                    if target == 'el':
                        if i == 0:
                            el_rms1.append(df_k2["Test RMS Compared"].mean())
                            el_std1.append(df_k2["Test RMS Compared"].std())
                        elif i == 1:
                            el_rms2.append(df_k2["Test RMS Compared"].mean())
                            el_std2.append(df_k2["Test RMS Compared"].std())

        df_output = pd.DataFrame({'k': k_list, 'Az RMS 1': az_rms1, 'Az STD 1': az_std1, 'El RMS 1': el_rms1, 'El STD 1': el_std1, 'Az RMS 2': az_rms2, 'Az STD 2': az_std2, 'El RMS 2': el_rms2, 'El STD 2': el_std2})
        path_output = os.path.join(descr_days, ds + '_results_table' '.csv')
        df_output.to_latex(path_output, index = False)
                



def make_table_cases(dfs, descr):


    for ds in dfs[0].dataset.unique():
        print(ds)
        k_list = []
        az_rms1 = []
        az_rms2 = []
        el_rms1 = []
        el_rms2 = []
        az_std1 = []
        az_std2 = []
        el_std1 = []
        el_std2 = []
        for i, df in enumerate(dfs):

            for target in df.target.unique():
                df_k = df[(df['dataset'] == ds) & (df['target'] == target)]
                #Mean val RMS compared and mean test RMS compared
                # print(f'Dataset: {ds.ljust(39)}, Target: {target}, Mean compared val RMS: {df_tmp["Val RMS Compared"].mean():.3f}, Mean compared test RMS: {df_tmp["Test RMS Compared"].mean():.3f}, STD: {df_tmp["Test RMS Compared"].std():.3f}')

                for k in df.num_features.unique():
                    df_k2 = df_k[df_k['num_features'] == k]
                    #Mean val RMS compared and mean test RMS compared
                    # print(f'k: {k}, Dataset: {ds.ljust(39)}, Target: {target}, Mean compared val RMS: {df_k2["Val RMS Compared"].mean():.3f}, Mean compared test RMS: {df_k2["Test RMS Compared"].mean():.3f}, STD: {df_k2["Test RMS Compared"].std():.3f}')
                    #Print k, target, mean compared test RMS and std, with not other text

                    if target == 'az':
                        if i == 0:
                            az_rms1.append(df_k2["Test RMS Compared"].mean())
                            az_std1.append(df_k2["Test RMS Compared"].std())
                            k_list.append(k)
                        elif i == 1:
                            az_rms2.append(df_k2["Test RMS Compared"].mean())
                            az_std2.append(df_k2["Test RMS Compared"].std())
                    
                    if target == 'el':
                        if i == 0:
                            el_rms1.append(df_k2["Test RMS Compared"].mean())
                            el_std1.append(df_k2["Test RMS Compared"].std())
                        elif i == 1:
                            el_rms2.append(df_k2["Test RMS Compared"].mean())
                            el_std2.append(df_k2["Test RMS Compared"].std())

        df_output = pd.DataFrame({'k': k_list, 'Az RMS 1': az_rms1, 'Az STD 1': az_std1, 'El RMS 1': el_rms1, 'El STD 1': el_std1, 'Az RMS 2': az_rms2, 'Az STD 2': az_std2, 'El RMS 2': el_rms2, 'El STD 2': el_std2})
        path_output = os.path.join(descr, ds +  f'_results_table.csv')
        df_output.to_latex(path_output, index = False, float_format="%.3f")
                 

def arguments_for_table_cases():
    df_short2 = pd.read_csv('./FinalResults/Run10/model_parameters_10.csv')
    df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')

    df_short = pd.read_csv('./FinalResults/Run6/model_parameters_6.csv')
    df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')

    # #Append df2's to the end of the originals
    df_short = df_short.append(df_short2)
    df_short_days = df_short_days.append(df_short_days2)


    df_all_days04 = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')
    df_all_random04 = pd.read_csv('./FinalResults/Run12/model_parameters_6.csv')
    df_all_days05 = pd.read_csv('./FinalResults/Run12/model_parameters_7.csv')
    df_all_random05 = pd.read_csv('./FinalResults/Run12/model_parameters_8.csv')


    randomly_split = [df_short, df_all_random04]
    split_on_days = [df_short_days, df_all_days04]

    
    descr_random = 'FinalTableRandomySplit_v2'
    descr_days = 'FinalTableSplitOnDays_v2'

    if not os.path.exists(descr_days):
        os.makedirs(descr_days)
    if not os.path.exists(descr_random):
        os.makedirs(descr_random)


    make_table_cases(randomly_split, descr_random)
    make_table_cases(split_on_days, descr_days)


def make_table_min_val(descr, dfs, fn=''):


    ds_list = []
    az_rms1 = []
    az_rms2 = []
    el_rms1 = []
    el_rms2 = []
    az_std1 = []
    az_std2 = []
    el_std1 = []
    el_std2 = []

    for i, df in enumerate(dfs):
        
        # df = df[df['fold'] == 5]
        min_val_idx = df.groupby(['dataset', 'fold', 'target'])['Val RMS Compared'].idxmin()
        min_val = df.loc[min_val_idx]


        for ds in dfs[0].dataset.unique():
            for target in df.target.unique():
                df_tmp = min_val[(min_val['dataset'] == ds) & (min_val['target'] == target)]

                if target == 'az':
                    if i == 0:
                        az_rms1.append(df_tmp["Test RMS Compared"].mean())
                        az_std1.append(df_tmp["Test RMS Compared"].std())
                        ds_list.append(ds)
                    elif i == 1:
                        az_rms2.append(df_tmp["Test RMS Compared"].mean())
                        az_std2.append(df_tmp["Test RMS Compared"].std())
                elif target == 'el':
                    if i == 0:
                        el_rms1.append(df_tmp["Test RMS Compared"].mean())
                        el_std1.append(df_tmp["Test RMS Compared"].std())
                    elif i == 1:
                        el_rms2.append(df_tmp["Test RMS Compared"].mean())
                        el_std2.append(df_tmp["Test RMS Compared"].std())

    

    df_output = pd.DataFrame({'Dataset': ds_list, 'Az RMS 1': az_rms1, 'Az STD 1': az_std1, 'El RMS 1': el_rms1, 'El STD 1': el_std1, 'Az RMS 2': az_rms2, 'Az STD 2': az_std2, 'El RMS 2': el_rms2, 'El STD 2': el_std2})
    path_output = os.path.join(descr, ds +  f'_results_table_minval{fn}.csv')
    df_output.to_latex(path_output, index = False, float_format="%.3f")


def arguments_for_table_min_val():

    df_short2 = pd.read_csv('./FinalResults/Run10/model_parameters_10.csv')
    df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')

    df_short = pd.read_csv('./FinalResults/Run6/model_parameters_6.csv')
    df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')

    # #Append df2's to the end of the originals
    df_short = df_short.append(df_short2)
    df_short_days = df_short_days.append(df_short_days2)


    df_all_days04 = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')
    df_all_random04 = pd.read_csv('./FinalResults/Run12/model_parameters_6.csv')
    df_all_days05 = pd.read_csv('./FinalResults/Run12/model_parameters_7.csv')
    df_all_random05 = pd.read_csv('./FinalResults/Run12/model_parameters_8.csv')


    dfs_random = [df_short, df_all_random04]
    dfs_days   = [df_short_days, df_all_days04]
    dfs_05     = [df_all_days05, df_all_random05]

    descr_random = 'FinalTableRandomySplit_v2'
    descr_days = 'FinalTableSplitOnDays_v2'

    if not os.path.exists(descr_days):
        os.makedirs(descr_days)
    if not os.path.exists(descr_random):
        os.makedirs(descr_random)


    # make_table_min_val(descr_random, dfs_random)
    make_table_min_val(descr_days, dfs_days)
    # make_table_min_val(descr_days, dfs_05, 'all_05')


def make_table_result_folds_hp():


    df_all_days04 = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')

   
    best_k = {'az': 2, 'el': 50}

    dataset_list = []
    target_list = []

    first = True
    for ds in ['tmp2022_clean_clf', 'tmp2022_clean_clf_nflash230']:
        for target in ['az', 'el']:
            for fold in range(6):
                df_tmp = df_all_days04[(df_all_days04['dataset'] == ds) & (df_all_days04['fold'] == fold) & (df_all_days04['target'] == target) & (df_all_days04['num_features'] == best_k[target])]
                try:
                    params = df_tmp['Model Parameters'].values[0].replace('\'', '\"')
                except:
                    embed()
                params = json.loads(params) #Dictionary of model parameters
                #Append hp to dataframe
                if first:
                    df_output = pd.DataFrame(params, index=[0])
                    first = False
                else:
                    df_output = df_output.append(params, ignore_index=True)
                
                dataset_list.append(ds)
                target_list.append(target)

    embed()
    df_output.insert(0, 'Dataset', dataset_list)
    df_output.insert(1, 'Target', target_list)
    print(df_output)
    df_output.to_latex('./FinalTableSplitOnDays_v2/hyperparams_selected.tex', index=False, float_format="%.3f"  )


def make_table_result_folds_rms_ds():

    df_all_days04 = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')

    df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')
    df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')
    df_short_days = df_short_days.append(df_short_days2)


    best_k = {'az': 2, 'el': 50}

    target_list = []
    val_rms1 = []
    val_rms2 = []
    test_rms1 = []
    test_rms2 = []

    first = True
    for i, ds in enumerate(['tmp2022_clean_clf', 'tmp2022_clean_clf_nflash230']):
        for target in ['az', 'el']:
            for fold in range(6):
                df_tmp = df_all_days04[(df_all_days04['dataset'] == ds) & (df_all_days04['fold'] == fold) & (df_all_days04['target'] == target) & (df_all_days04['num_features'] == best_k[target])]


                if i == 0:
                    val_rms1.append(df_tmp['Val RMS Compared'].values[0])
                    test_rms1.append(df_tmp['Test RMS Compared'].values[0])
                    target_list.append(target)
                
                elif i == 1:
                    val_rms2.append(df_tmp['Val RMS Compared'].values[0])
                    test_rms2.append(df_tmp['Test RMS Compared'].values[0])

                                

    embed()
    
    df_output = pd.DataFrame({
        'Target': target_list,
        'Val RMS 1': val_rms1,
        'Test RMS 1': test_rms1,
        'Val RMS 2': val_rms2,
        'Test RMS 2': test_rms2
    })
    df_output.to_latex('./FinalTableSplitOnDays_v2/selected_rms_folds.tex', index=False, float_format="%.3f")

def make_table_result_folds_rms_df():

    df_all_days04 = pd.read_csv('./FinalResults/Run12/model_parameters_5.csv')

    df_short_days2 = pd.read_csv('./FinalResults/Run10/model_parameters_11.csv')
    df_short_days = pd.read_csv('./FinalResults/Run7/model_parameters_7.csv')
    df_short_days = df_short_days.append(df_short_days2)



    target_list = []
    val_rms1 = []
    val_rms2 = []
    test_rms1 = []
    test_rms2 = []
    k_az = []
    k_el = []

    ds = 'tmp2022_clean_clf'
    ds = 'tmp2022_clean_clf_nflash230'

    first = True
    for i, df in enumerate([df_short_days, df_all_days04]):
        
        df = df[(df['dataset'] == ds)]
        min_val_idx = df.groupby(['dataset', 'fold', 'target'])['Val RMS Compared'].idxmin()
        min_val = df.loc[min_val_idx]

        for target in ['az', 'el']:
            for fold in range(6):
                df_tmp = min_val[(min_val['dataset'] == ds) & (min_val['target'] == target) & (min_val['fold'] == fold)]
                # df_tmp = df_all_days04[(df_all_days04['dataset'] == ds) & (df_all_days04['fold'] == fold) & (df_all_days04['target'] == target) & (df_all_days04['num_features'] == best_k[target])]

                if i == 0:
                    val_rms1.append(df_tmp['Val RMS Compared'].values[0])
                    test_rms1.append(df_tmp['Test RMS Compared'].values[0])
                    k_az.append(df_tmp['num_features'].values[0])
                    target_list.append(target)
                
                elif i == 1:
                    val_rms2.append(df_tmp['Val RMS Compared'].values[0])
                    test_rms2.append(df_tmp['Test RMS Compared'].values[0])
                    k_el.append(df_tmp['num_features'].values[0])
                                

    embed()
    
    df_output = pd.DataFrame({
        'Target': target_list,
        'k az': k_az,
        'Val RMS 1': val_rms1,
        'Test RMS 1': test_rms1,
        'k el': k_el,
        'Val RMS 2': val_rms2,
        'Test RMS 2': test_rms2
    })
    df_output.to_latex('./FinalTableSplitOnDays_v2/rms_folds_minval_cases_n230_wk.tex', index=False, float_format="%.3f")

from dataset import PrepareDataCombined


def linreg_rawdata_cv(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)


    nonlinear_features = []
    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']
    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5']#'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    az_features = ['HASA', 'HSCA2', 'HACA3', 'HASA2', 'HACA2', 'HSCA', 'HSCA5',
                    'C', 'COMMANDAZ', 'COMMANDEL']
    el_features = ['HESE', 'HECE', 'HACA2', 'HASA2', 'HACA3', 'HESA3', 'HESA4',
                    'HESA5', 'C', 'COMMANDEL']

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)

    rms_train_az = np.zeros(n_folds)
    rms_train_el = np.zeros(n_folds)
    rms_train_tot = np.zeros(n_folds)

    rms_test_az = np.zeros(n_folds)
    rms_test_el = np.zeros(n_folds)
    rms_test_tot = np.zeros(n_folds)

    for j in range(n_folds):
        print(f'Fold {j+1}/{n_folds}')
        
        df_test = df.iloc[split_indices[j] : split_indices[j+1]]
        df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 
       
        ds = PrepareDataCombined(
            df = df_trainval,
            nonlinear_features = az_features,
            linear_features = el_features,
            scale_data = False
        )


        #Concatenate features together, ds.X_linear_train and ds.X_nonlinear_train
        X_train_az = ds.X_nonlinear_train
        X_train_el = ds.X_linear_train

        X_val_az = ds.X_nonlinear_test
        X_val_el = ds.X_linear_test
        y_train = np.rad2deg(ds.y_train) * 3600
        y_val = np.rad2deg(ds.y_test) * 3600

        # X_train_az = df_trainval[az_features].values
        # X_train_el = df_trainval[el_features].values
        # y_train = np.rad2deg(df_trainval[['OFFSETAZ', 'OFFSETEL']].values) * 3600
        
        X_test_az = df_test[az_features].values
        X_test_el = df_test[el_features].values
        y_test = np.rad2deg(df_test[['OFFSETAZ', 'OFFSETEL']].values) * 3600


        model_az = LinearRegressor(target=f'az_fold{j}')
        model_az.fit(X_train_az, y_train[:,0])

        model_el = LinearRegressor(target=f'el_fold{j}')
        model_el.fit(X_train_el, y_train[:,1])


        y_pred_az = model_az.predict(X_train_az)
        y_pred_el = model_el.predict(X_train_el)

        rms_train_az[j] = np.sqrt( mean_squared_error(y_train[:,0], y_pred_az) )
        rms_train_el[j] = np.sqrt( mean_squared_error(y_train[:,1], y_pred_el) )
        rms_train_tot[j] = np.sqrt( np.mean( np.linalg.norm(y_train - np.stack([y_pred_az, y_pred_el], axis=1), axis=1)**2 ) )
        print(f'AZ RMS: {rms_train_az[j]:.3f}, EL RMS: {rms_train_el[j]:.3f}, Total RMS: {rms_train_tot[j]:.3f}')
        
        y_pred_az = model_az.predict(X_test_az)
        y_pred_el = model_el.predict(X_test_el)

        rms_test_az[j] = np.sqrt( mean_squared_error(y_test[:,0], y_pred_az) )
        rms_test_el[j] = np.sqrt( mean_squared_error(y_test[:,1], y_pred_el) )
        rms_test_tot[j] = np.sqrt( np.mean( np.linalg.norm(y_test - np.stack([y_pred_az, y_pred_el], axis=1), axis=1)**2 ) )

        print(f'AZ RMS: {rms_test_az[j]:.3f}, EL RMS: {rms_test_el[j]:.3f}, Total RMS: {rms_test_tot[j]:.3f}')


    df_results = pd.DataFrame({
        'Fold': np.arange(n_folds),
        'RMS Train Az': rms_train_az,
        'RMS Train El': rms_train_el,
        'RMS Train': rms_train_tot,
        'RMS Test Az': rms_test_az,
        'RMS Test El': rms_test_el,
        'RMS Test': rms_test_tot
    })
    
    df_results.to_csv(f'{PATH_RESULTS}linreg_rn{run_number}.csv', index=False)
    print(df_results.mean())
    embed()

def compare_results_transformed(process_number = 99, run_number = 99, split_on_days = False):
    

    path_results = f'./FinalResults/Run{run_number}/' 
    for _path in [path_results, path_results + 'SelectedFeatures/']:
        if not os.path.exists(_path):
            os.makedirs(_path)


    
    path_datasets = './Datasets/'
    datasets = [(4, 'tmp2022_clean_clf_transformed'), (6, 'tmp2022_clean_clf_nflash230_transformed')]

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']

    # create an empty DataFrame
    df_model_params = pd.DataFrame(columns=['dataset', 'fold','target','num_features', 'Model Name', 'Model Parameters', 'Val RMS Model', 'Val RMS Current', 'Val RMS Compared', 'Test RMS Model', 'Test RMS Current', 'Test RMS Compared'])

    n_folds = 6

    for dataset_key,dataset in datasets:
        df = pd.read_csv(os.path.join(path_datasets, f'{dataset}/features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(axis=1, thresh=len(df)-67, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df_benchmark = pd.read_csv(path_datasets + dataset + '.csv')
        df_benchmark['obs_date'] = pd.to_datetime(df_benchmark['obs_date'])
        df_benchmark.sort_values(by='obs_date', inplace=True)
        if dataset_key == 4:
            df_benchmark = df_benchmark.loc[df_benchmark['obs_date'].isin(df['date'])]
        elif dataset_key == 6:
            df_benchmark = df_benchmark.loc[df_benchmark['obs_date'].isin(df['date'])]
            df_benchmark = df_benchmark.loc[df_benchmark['rx'] == 'NFLASH230']

        n = len(df)
        #List on split sizes, n%n_folds with length n//n_folds + 1 and n_folds - n%n_folds with length n//n_folds, but each on is also the sum of the list until that point
        split_indices = get_split_indices(n, n_folds)

        for i in range(n_folds):
            df_test = df.iloc[split_indices[i] : split_indices[i+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[i]], df.iloc[split_indices[i+1]:]])

            df_benchmark_test = df_benchmark.iloc[split_indices[i] : split_indices[i+1]]

            # embed(header = 'first loop xgb cv all')
            embed()


def replot_for_xgb_exp(process_number = 99, run_number = 99, split_on_days = False):
    

    path_results = f'./FinalResults/Run{run_number}/' 
    for _path in [path_results, path_results + 'SelectedFeatures/']:
        if not os.path.exists(_path):
            os.makedirs(_path)


    
    path_datasets = './Datasets/'
    datasets = [(2,'tmp2022_clean_clf_nflash230'), (1,'tmp2022_clean_clf')]#, (4, 'tmp2022_clean_clf_transformed'), (6, 'tmp2022_clean_clf_nflash230_transformed')]

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']

    # create an empty DataFrame
    df_model_params = pd.DataFrame(columns=['dataset', 'fold','target','num_features', 'Model Name', 'Model Parameters', 'Val RMS Model', 'Val RMS Current', 'Val RMS Compared', 'Test RMS Model', 'Test RMS Current', 'Test RMS Compared'])

    n_folds = 6

    for dataset_key,dataset in datasets:
        df = pd.read_csv(os.path.join(path_datasets, f'{dataset}/features_offsets.csv'))
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(axis=1, thresh=len(df)-67, inplace=True)
        df.dropna(inplace=True)

        n = len(df)
        #List on split sizes, n%n_folds with length n//n_folds + 1 and n_folds - n%n_folds with length n//n_folds, but each on is also the sum of the list until that point
        split_indices = get_split_indices(n, n_folds)

        for i in range(n_folds):
            df_test = df.iloc[split_indices[i] : split_indices[i+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[i]], df.iloc[split_indices[i+1]:]])

            # embed(header = 'first loop xgb cv all')
            for target_key in ['az', 'el']:
                for k in [2,5,10,20,30,40,50]:      
                    for remove_correlated in [True]:
                        model_name = f'XGB_ds{dataset_key}_tp{i}_k{k}_{"uncorr" if remove_correlated else "corr"}_{target_key}'

                        feature_list = get_selected_features(df_trainval[~df_trainval.isin(constant_features)].copy(), k, target_key, remove_correlated, path_results, model_name)
                        feature_list += constant_features
                        feature_list = list(dict.fromkeys(feature_list))

                        if len(df_trainval['rx'].unique()) > 1:
                            feature_list.append('rx')

                        ds = PrepareDataFinal(
                            df = df_trainval.copy(),
                            parameter_keys = (process_number, dataset_key),
                            feature_list = feature_list,
                            target_key = target_key,
                            run_number = run_number,
                            split_on_days = split_on_days
                        )
                        #Test model on df_test
                        X = df_test[feature_list]
                        if 'rx' in X.columns:
                            le = LabelEncoder()
                            X['rx'] = le.fit_transform(X['rx'])
                        X = X.values
                        y = df_test['Off_Az' if target_key == 'az' else 'Off_El'].values

                        y_pred = model.model.predict(X)

                        RMS_test_model = np.sqrt(mean_squared_error(y, y_pred))
                        RMS_test_current = np.sqrt(mean_squared_error(y, np.zeros_like(y)))
                        RMS_test_compared = RMS_test_model / RMS_test_current

                        model = XGBoostRegressor(ds, name = model_name, train_model=False, load_model=True)
                        model.plot_histogram(X = X, y = y)
                        model.plot_sorted_prediction_final(X = X, y = y)
                        model.plot_histogram()
                        model.plot_sorted_prediction_final()

                    

                    # save the DataFrame to a CSV file
                    # df_model_params.to_csv(path_results + f'model_parameters_{process_number}.csv', index=False)



def SAGE(model, X, y, features, path_save):

    size = 512
    imputer = sage.MarginalImputer(model, X[:size])
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_testues = estimator(X[:size], y[:size])


    sage_testues.plot(feature_names=features, figsize = (12,8), label_size = 16, title = 'Global feature importance')
    plt.tight_layout()

    embed()
    plt.savefig(path_save.replace('png','pdf'), dpi = 300)

def fit_xgb_model(X,y,parameters):
    #Function to fit a model with given parameters and data
    model = xgb.XGBRegressor(**parameters)
    model.fit(X, y)

    return model


def get_xgb_result_path(run_number, process_number):
    return f'./FinalResults/Run{run_number}/model_parameters_{process_number}.csv'


def return_model_params(path,name, fold):
    df = pd.read_csv(path)

    params = df.loc[(df['Model Name'] == name) & (df['fold'] == fold)]['Model Parameters'].values[0]
    params = json.loads(params.replace('\'', '\"'))

    return params



def run_sage_specific_models(run_number, process_number):
    path_sage = f'./FinalResults/Run{run_number}/Plots/SAGE/'
    path_results = f'./FinalResults/Run{run_number}/' 
    n_folds = 6
    path_dataset = './Datasets/tmp2022_clean_clf_nflash230/features_offsets.csv'
    df = pd.read_csv(path_dataset)
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(axis=1, thresh=len(df)-67, inplace=True)
    df.dropna(inplace=True)

    constant_features = ['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']
    n = len(df)
    split_indices = get_split_indices(n, n_folds)

    models = [('az', 2, 30), ('el', 2, 40)]

    i = 5

    df_test = df.iloc[split_indices[i] : split_indices[i+1]]
    df_trainval = pd.concat([df.iloc[:split_indices[i]], df.iloc[split_indices[i+1]:]])

    remove_correlated = True
    for target_key, dataset_key, k in models:
        model_name = f'XGB_ds{dataset_key}_tp{i}_k{k}_{"uncorr" if remove_correlated else "corr"}_{target_key}'

        feature_list = get_selected_features(df_trainval[~df_trainval.isin(constant_features)].copy(), k, target_key, remove_correlated, path_results, model_name)
        feature_list += constant_features
        feature_list = list(dict.fromkeys(feature_list))

        if len(df_trainval['rx'].unique()) > 1:
            feature_list.append('rx')

        ds = PrepareDataFinal(
            df = df_trainval.copy(),
            parameter_keys = (process_number, dataset_key),
            feature_list = feature_list,
            target_key = target_key,
            run_number = run_number,
            split_on_days = True
        )


        X_train, X_val, y_train, y_val = ds.get_data()
        X_test = df_test[feature_list].values
        y_test = df_test['Off_Az' if target_key == 'az' else 'Off_El'].values

        

        print('Getting path')
        path   = get_xgb_result_path(run_number, process_number)
        print('Getting params')
        params = return_model_params(path,model_name,i)
        print('Fitting model')
        model  = fit_xgb_model(X_train, y_train, params)

        #Save model with pickle
        with open(f'{path_results}Models/{model_name}_nflash230.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        #Calculate val and test rms
        y_pred = model.predict(X_val)
        RMS_val_model = np.sqrt(mean_squared_error(y_val, y_pred))
        RMS_val_current = np.sqrt(mean_squared_error(y_val, np.zeros_like(y_val)))
        RMS_val_compared = RMS_val_model / RMS_val_current

        y_pred = model.predict(X_test)
        RMS_test_model = np.sqrt(mean_squared_error(y_test, y_pred))
        RMS_test_current = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))
        RMS_test_compared = RMS_test_model / RMS_test_current

        print(f'Val RMS Model: {RMS_val_model:.3f}, Val RMS Current: {RMS_val_current:.3f}, Val RMS Compared: {RMS_val_compared:.3f}')
        print(f'Test RMS Model: {RMS_test_model:.3f}, Test RMS Current: {RMS_test_current:.3f}, Test RMS Compared: {RMS_test_compared:.3f}')



        model_object = XGBoostRegressor(ds, name = model_name, train_model=False, load_model=True, model_path = f'{path_results}Models/{model_name}_nflash230.pkl')
         
        model_object.plot_histogram(X = X_val.values, y = y_val.values, fn = '_val')
        model_object.plot_sorted_prediction_final(X = X_val.values, y = y_val.values, fn = '_val')

        model_object.plot_histogram(X = X_test, y = y_test, fn = '_test')
        model_object.plot_sorted_prediction_final(X = X_test, y = y_test, fn = '_test')

        # print('SAGE training set')
        # SAGE(model, X_train.values, y_train.values, feature_list, path_save = path_sage + f'{model_name}_train.png')
        # print('SAGE validation set')
        # SAGE(model, X_val.values, y_val.values, feature_list, path_save = path_sage + f'{model_name}_val.png')
        # print('SAGE test set')
        # SAGE(model, X_test, y_test, feature_list, path_save = path_sage + f'{model_name}_test.png')




if __name__ == '__main__':




    # data = PrepareData()
    # model = LightGBM(data)
    # model.plot_sorted_predictions()
    # print_final_results_from_offset_prediction()
    # model = XGBoostRegressor(data)
    # model.plot_sorted_predictions()
    # XGB_experiment_CV_all()
    # for i in range(5):
    #     fit_linear_models(i) 
    # nohup python models_v2.py > out1.log 2> error1.log &
    # nohup python models_v2.py > out5.log 2> error5.log &
    # nohup python models_v2.py > out6.log 2> error6.log &
    # nohup python models_v2.py > out7.log 2> error7.log &
    # nohup python models_v2.py > out8.log 2> error8.log &
    # nohup python models_v2.py > out4.log 2> error4.log &

    # XGB_experiment_CV_short(process_number = 1) 
    # XGB_experiment_CV_short(process_number = 2)  
    # XGB_experiment_CV_all(process_number = 5, run_number = 12, split_on_days=True) #days ts 0.4 beehive 20 command
    # XGB_experiment_CV_all(process_number = 6, run_number = 12, split_on_days=False) #random ts 0.4 bh 21    command
    # XGB_experiment_CV_all(process_number = 7, run_number = 12, split_on_days=True) #days ts 0.5 beehive 22     command
    # XGB_experiment_CV_all(process_number = 8, run_number = 12, split_on_days=False) #random ts 0.5 bh 23        command
    # XGB_experiment_CV_all(process_number = 3) #actualaz days
    # XGB_experiment_CV_all(process_number = 4) #actualel random


    # df0 = pd.read_csv('/mn/stornext/d17/extragalactic/personal/bendikny/code/FinalResultsOptical/Run1/results_tp0.csv')
    # df1 = pd.read_csv('/mn/stornext/d17/extragalactic/personal/bendikny/code/FinalResultsOptical/Run1/results_tp1.csv')
    # df2 = pd.read_csv('/mn/stornext/d17/extragalactic/personal/bendikny/code/FinalResultsOptical/Run1/results_tp2.csv')
    # df3 = pd.read_csv('/mn/stornext/d17/extragalactic/personal/bendikny/code/FinalResultsOptical/Run1/results_tp3.csv')
    # df4 = pd.read_csv('/mn/stornext/d17/extragalactic/personal/bendikny/code/FinalResultsOptical/Run1/results_tp4.csv')
    # linreg_raw_data_for_comparison()
    # analyze_results_corrections()
    # analyze_results_cases()
    #model = XGBoost(data)
    #model.plot_sorted_predictions()
    # add_model_output_to_dataset(name = 'optical_optimal_ts01', path_dataset = './Data/scans_nflash230_unscaled_all.csv', new_name = 'all_ts01')
    # add_xgb_model_output_to_dataset(path_dataset = './Data/scans_nflash230_unscaled_all.csv')
    # add_linreg_output_to_dataset(path_dataset = './Data/scans_nflash230_unscaled_all.csv')
    # xgb_linear_terms()
    # XGB_Optical()
    # test_xgb_on_pointing_scans()
    #model = RandomForest(data)
    # new_model_on_pointing_scans()
    # fit_linear_models()
    # test_linreg_on_pointing_scans()
    # analyze_results_corrections_k()
    # arguments_for_table_min_val()
    # make_table_result_folds_hp()
    # make_table_result_folds_rms()
    # arguments_for_table_cases()
    # arguments_for_table_min_val()
    # make_table_result_folds_rms_df()
    # linreg_rawdata_cv(run_number = 6)
    # data_clfScans = PointingScansClassification_v2('./Data/tmp2022_cleanedRules.csv')
    # model = XGBoostClassifier(data_clfScans, train_model = False, load_model=True)
    # compare_results_transformed()
    run_sage_specific_models(run_number = 12, process_number = 5)
    
    print('run completed')
    # df = pd.read_csv('./Datasets/tmp2022_clean_nflash230_transformed/features_offsets.csv')
    
    # ignore_feats = ['date', 'rx', 'ca', 'ie', 'Off_Az', 'Off_El', 'hours_since_corr', 'month_continuous', 'month', 'time_of_day']
    # features = df.loc[: , ~df.columns.isin(ignore_feats)].columns
    # embed()
    # feature_name = [split[0] for split in features.str.split('_')]
    # feature_type = [split[1] for split in features.str.split('_')]

    feats = ['date', 'WINDSPEED_VAR_5', 'WINDDIRECTION_MEDIAN_1', 'SUNAZ_MEDIAN_1', 'SUNEL_MEDIAN_1',
       'TILT1T_MEDIAN_1', 'TILT1X_MEDIAN_1', 'TILT1Y_MEDIAN_1',
       'TEMP1_MEDIAN_1', 'TEMP26_MEDIAN_1', 'TEMPERATURE_MEDIAN_1',
       'POSITIONX_MEDIAN_1', 'POSITIONY_MEDIAN_1', 'POSITIONZ_MEDIAN_1',
       'ROTATIONX_MEDIAN_1', 'ROTATIONY_MEDIAN_1', 'HUMIDITY_MEDIAN_1',
       'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1', 'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1',
       'DISP_ABS3_MEDIAN_1']

    """

    data_analytical_v3 = PrepareDataAnalytical_v3(target_key = 'testaz', use_time_filter = False)
    model = XGBoostRegressor(data_analytical_v3, name = 'XGB_analytical_el_75_all_freq_subca', train_model = True)
    model.plot_sorted_predictions_v2() 
    model.plot_error_locations()

    data_analytical_v3 = PrepareDataAnalytical_v3(target_key = 'testel', use_time_filter = False)
    model = XGBoostRegressor(data_analytical_v3, name = 'XGB_analytical_az_75_all_freq_subie', train_model = True)
    model.plot_sorted_predictions_v2()
    model.plot_error_locations()

    data_analytical_v2 = PrepareDataAnalytical_v2(target_key = 'az', use_time_filter = True)
    model = XGBoostRegressor(data_analytical_v2, name = 'XGB_analytical_az', train_model = True)
    model.plot_sorted_predictions()

    data_analytical_v2 = PrepareDataAnalytical_v2(target_key = 'el', use_time_filter = True)
    model = XGBoostRegressor(data_analytical_v2, name = 'XGB_analytical_el', train_model = True)
    model.plot_sorted_predictions()

    data_analytical = PrepareDataAnalytical(target = 'ACTUALAZ_MEDIAN')
    model = XGBoostRegressor(data_analytical, name = 'XGB_analytical_az', train_model = True)
    model.plot_sorted_predictions()

    data_analytical = PrepareDataAnalytical()
    model = XGBoostRegressor(data_analytical, name = 'XGB_analytical_el', train_model = True)
    model.plot_sorted_predictions()
    data_clfScans = PointingScansClassification_v2('./Data/tmp2022_cleanedRules.csv')
    model = XGBoostClassifier(data_clfScans, train_model = False)
    SAGE()

    data_clfScans = PointingScansClassification('./Data/tmp2022.csv', var = 'el')
    model = XGBoostClassifier(data_clfScans, train_model = True)

    data_clfScans = PointingScansClassification('./Data/tmp2022.csv', var = 'az')
    model = XGBoostClassifier(data_clfScans, train_model = True)"""

"""
    
    for key_dataset, dataset_path in dataset_tests.items():
        print(f'Dataset {key_dataset} | {datasets[key_dataset]}...')
        df = pd.read_csv(dataset_path)
        df['date'] = pd.to_datetime(df['date'])

        for key_timeperiod, timeperiod_tuple in time_period_tests.items():
            print(f'Time Period {key_timeperiod} | {timeperiod_tuple}...')
            for key_feature, feature_list in feature_tests.items():
                print(f'Feature {key_feature}...')
                
                for target_key in ['az', 'el']:
                    for k in [2,3,5,10,15,20,25,30,40,50]:
                        df_tmp = df[(df['date'] >= timeperiod_tuple[0]) & (df['date'] <= timeperiod_tuple[1])]

                        selected_features = ....
                        selected_features_not_correlated = ...
                        
                        df_tmp = df_tmp[selected_features_not_correlated]


import numpy as np

for key_dataset, dataset_path in dataset_tests.items():
    print(f'Dataset {key_dataset} | {datasets[key_dataset]}...')
    df = pd.read_csv(dataset_path)
    df['date'] = pd.to_datetime(df['date'])

    for key_timeperiod, timeperiod_tuple in time_period_tests.items():
        print(f'Time Period {key_timeperiod} | {timeperiod_tuple}...')
        for key_feature, feature_list in feature_tests.items():
            print(f'Feature {key_feature}...')
            
            for target_key in ['az', 'el']:
                for k in [2,3,5,10,15,20,25,30,40,50]:
                    for remove_correlated in [True, False]:
                        df_tmp = df[(df['date'] >= timeperiod_tuple[0]) & (df['date'] <= timeperiod_tuple[1])]

                        # Feature selection using mutual information
                        X = df_tmp[feature_list]
                        y = df_tmp[target_key]
                        selector = SelectKBest(mutual_info_regression, k=k)
                        selector.fit(X, y)
                        selected_features = X.columns[selector.get_support()]

                        if remove_correlated:
                            # Removal of highly correlated features
                            selected_features_not_correlated = []
                            corr_matrix = np.abs(df_tmp[selected_features].corr())
                            for i in range(len(selected_features)):
                                if selected_features[i] not in selected_features_not_correlated:
                                    corr_values = corr_matrix[selected_features[i]][selected_features].values
                                    idx = np.argwhere(corr_values > 0.75)
                                    if len(idx) > 1:
                                        mean_corr = np.mean(corr_values[idx])
                                        for j in idx:
                                            if corr_values[j] >= mean_corr:
                                                selected_features_not_correlated.append(selected_features[j])
                                    else:
                                        selected_features_not_correlated.append(selected_features[i])
                            selected_features = selected_features_not_correlated

                            
                        df_tmp = df_tmp[selected_features]
"""