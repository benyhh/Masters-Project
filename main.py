import importlib
import functions
import arguments
import data_analysis
import data_processing
import models

importlib.reload(functions)    ; importlib.reload(arguments); importlib.reload(models)
importlib.reload(data_analysis); importlib.reload(data_processing)

import models
import data_analysis
import data_processing
import arguments
from functions import *
from functions import random_seed

random.seed(random_seed)
np.random.seed(random_seed)

from models import feature_lists
from models import patches




def grid_models():
    PATH_FEATURES = './Data/merged_features3_all.csv'
    mods = ['RandomForest']
    targs = ['az', 'el']

    idx = 0
    for m in mods:
        df_mse = pd.DataFrame() # MSE, MODEL, TARGET, FEATURES, PATCH
        for t in targs:
            if t == 'both' and m == 'RandomForest':
                continue

            for col_key in models.feature_lists.keys():
                for patch_key in patches.keys():

                    row = {}
                        
                    model = models.Model(
                            df_path              = PATH_FEATURES,
                            method               = m, 
                            target               = t,
                            load_model           = True,
                            selected_columns_key = col_key,
                            patch_key            = patch_key,
                            use_pca              = True
                            )
                    

                    if not hasattr(model, 'model'):
                        model.train(save=True)
                    
                    
                    model.plot_sorted_predictions()
                    
                    #model.SAGE()

                    row['mse']        = model.evaluate()
                    row['method']     = m
                    row['target']     = t
                    row['features']   = col_key
                    row['patch']      = patch_key
                    
                    for key,val in model.params.items():
                        row[key] = val
                    
                    try:
                        row = pd.DataFrame ( row, index = [0] )
                    except:
                        row = pd.DataFrame ( row )

                    df_mse = pd.concat( [df_mse, row], ignore_index=True)
                    

        df_mse.to_csv(f'./Results/Results_{m}_PCA_1612_2.csv', index = False)


def run(args):

    #Run data models
    if args.models:
        print("Running models")

        model = models.Model(
                df_path              = './Data/merged_features3_all.csv',
                method               = args.model,
                target               = args.target,
                load_model           = args.load,
                selected_columns_key = args.column_filter,
                patch_key            = args.patch_filter,
                use_pca              = args.PCA
                )

        if args.train:
            model.train(save = args.save)    
        
        if args.plot_sorted:
            print(f'Plot sorted = {args.plot_sorted}')
            model.plot_sorted_predictions()
        
        if args.evaluate:
            model.evaluate()
        
        if args.SAGE:
            model.SAGE()

        if args.SHAP:
            model.SHAP()

    #Run grid models
    if args.grid_model:
        grid_models()

    #Run data analysis
    if args.data_analysis:
        print("Running data analysis")

        analyis = data_analysis.Analysis(
                path_df    = args.path_data_analysis,
                patch_key  = args.patch_filter       ,
                instrument = args.instrument     
                )
        
        if args.correlation:
            analyis.plot_correlation()
        
        if args.pairplot:
            analyis.plot_pairs()

        if args.plotscans:
            analyis.scan_locations()
        

    #Run data processing
    if args.data_processing:
        print("Running data processing")

        data_processing = data_processing.DataProcessing(path_df = './Data/all_scans_15_3.csv')


params_filter = ['gamma','reg_alpha','reg_lambda','learning_rate','max_delta_step','max_depth','min_child_weight',
                 'missing','monotone_constraints','n_estimators','num_parallel_treescale_pos_weight','subsample']

correlated_cols = ['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
                   'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
                   'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                   'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','ROTATIONX_sumdabs2','ACTUALAZ_sumdabs2',
                   'TILT1X_sumdabs2','ACTUALEL_sumdabs5','POSITIONX_sumdabs5','ROTATIONX_sumdabs5']

print(len(correlated_cols), len(set(correlated_cols)))

df = pd.read_csv('./Data/PointingTable.csv')
df.insert(0,'Offset', np.sqrt(df['Off_El']**2 + df['Off_Az']**2) )
print(df['Off_Az'].abs().mean(), df['Off_El'].abs().mean(), df['Offset'].mean())

def evaluate_models():

    model_az = models.Model(
                df_path              = './Data/merged_features3_all.csv',
                method               = 'RandomForest',
                target               = 'az',
                load_model           = True,
                selected_columns_key = 'Corr_reduced2',
                patch_key            = 0,
                use_pca              = True
                )

    model_el = models.Model(
                df_path              = './Data/merged_features3_all.csv',
                method               = 'RandomForest',
                target               = 'el',
                load_model           = True,
                selected_columns_key = 'Corr_reduced3',
                patch_key            = 0,
                use_pca              = True
                )

    #assert(model_az.X_test, model_el.X_test), "X_test not the same for both models"
    testaz = model_az.df_test.loc[ model_az.df_test['date'].isin(model_el.df_test['date'].unique()) ]
    testel = model_el.df_test.loc[ model_el.df_test['date'].isin(testaz['date'].unique()) ]
    X_test_az, y_test_az = model_az.split_df(testaz, target = ['Off_Az'])
    X_test_el, y_test_el = model_el.split_df(testel, target = ['Off_El'])
    
    X_test_az, _ = model_az.PCA(X_test_az, X_test_az, loaded = True)
    X_test_el, _ = model_el.PCA(X_test_el, X_test_el, loaded = True)
    
    prediction_az = model_az.model.predict(X_test_az)
    prediction_el = model_el.model.predict(X_test_el)


    prediction_az = model_az.scaler2.inverse_transform(prediction_az.reshape(-1,1)).ravel()
    prediction_el = model_el.scaler2.inverse_transform(prediction_el.reshape(-1,1)).ravel()

    y_test_az     = model_az.scaler2.inverse_transform(y_test_az.values[:,0].reshape(-1,1)).ravel()
    y_test_el     = model_el.scaler2.inverse_transform(y_test_el.values[:,0].reshape(-1,1)).ravel()

    error_az   = y_test_az - prediction_az
    error_el   = y_test_el - prediction_el
    #mean_error = np.mean(np.sqrt(error_az**2 + error_el**2))

    #nopred = np.mean(np.sqrt(y_test_az.values**2 + y_test_el.values**2))
    #print(mean_error, nopred)
    
    print(np.mean(np.abs(error_az)), np.mean(np.abs(error_el)))
    print(np.mean(np.abs(y_test_az)), np.mean(np.abs(y_test_el)))

if __name__ == "__main__":
    # Parse arguments
    args = arguments.parser.parse_args()
    print(args)
    run(args)
    df_results  = pd.read_csv('./Results/Results_RandomForest_PCA_1612.csv')
    df_results2 = pd.read_csv('./Results/Results_RandomForest_PCA_1612_2.csv') 
    df_concat   = pd.concat([df_results, df_results2])
    df_concat = df_concat.sort_values(by=['mse'])
    #print(df_concat)
    #embed()
    # df_results = pd.read_csv('./Results/Results_RandomForest_PCA_d.csv')
    # df_results = df_results.sort_values(by=['mse'])
    # print(df_results)
    # embed()
    # df_results = pd.read_csv('./Results/Results_NN.csv')
    # df_results = df_results.sort_values(by=['mse'])
    # print(df_results)
    # embed()
    # df_results = pd.read_csv('./Results/Results_XGBoost.csv')
    # df_results = df_results.sort_values(by=['mse'])
    # print(df_results)quit
    # embed()



    # print(df_results)    
    #df = pd.read_csv('./Data/PointingTable.csv')
    #df['obs_date'] = pd.to_datetime(df['obs_date'])
    #print(df.loc[:, ['obs_date', 'scan', 'ie', 'ca', 'Off_Az', 'Off_El']])
    #print(df['rx'].value_counts())
    # for param in params_filter:
    #     if not param in df_results.columns:
    #         print(param)
    
    # print(df_results.reindex(columns = params_filter))
    # print(df_results.loc[ : , params_filter])
    

