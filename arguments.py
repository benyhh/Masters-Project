import argparse

# Arguments for processing, analysis, and models
parser = argparse.ArgumentParser(description='Model Class Arguments')

# Arguments for what scripts to utilize
parser.add_argument('--models'         , '-ms', action  = 'store_true', help = 'Run models')
parser.add_argument('--data_analysis'  , '-da', action  = 'store_true', help = 'Run data analysis')
parser.add_argument('--data_processing', '-dp', action  = 'store_true', help = 'Run data processing')
parser.add_argument('--grid_model'     , '-gm', action  = 'store_true', help = 'Run grid search function')


# Arguments for data processing
parser.add_argument('--path_data_processing', '-pdp', default = './Data/all_scans_15_3.csv',  type = str, help = 'Path to data for processing')

# Arguments for data analysis
parser.add_argument('--path_data_analysis', '-pda', default = None , type = str , help = 'Patch to data for analysis')
parser.add_argument('--correlation'       , '-c' ,  action  = 'store_true', help = 'Plot correlation or not')
parser.add_argument('--pairplot'          , '-pp',  action  = 'store_true', help = 'Plot data pairs')
parser.add_argument('--plotscans'         , '-ps',  action  = 'store_true', help = 'Plot scan locations')

# Model argument
parser.add_argument('--model'          , '-m' ,    default = 'RandomForest'    , type=str , help = 'Model to use')
parser.add_argument('--column_filter'  , '-cf',    default = 'Corr_reduced'        , type=str , help = 'Column filter to use')
parser.add_argument('--target'         , '-tg',    default = 'az'         , type=str , help = 'Target(s) for model')

parser.add_argument('--train'          , '-t' ,    action  = 'store_true' , help = 'Whether to train model or not')
parser.add_argument('--save'           , '-s' ,    action  = 'store_true' , help = 'Whether to save model or not')
parser.add_argument('--load'           , '-l' ,    action  = 'store_false', help = 'Whether to load model or not')
parser.add_argument('--plot_sorted'    , '-psp',   action  = 'store_true' , help = 'Whether to plot sorted predictions or not')
parser.add_argument('--evaluate'       , '-e' ,    action  = 'store_false', help = 'Whether to evaluate model or not')
parser.add_argument('--SHAP'           , '-shap',  action  = 'store_true' , help = 'Plot SHAP values')
parser.add_argument('--SAGE'           , '-sage',  action  = 'store_true' , help = 'Plot SAGE values')
parser.add_argument('--PCA'            , '-pca',   action  = 'store_true' , help = 'Do PCA transform')



# Common arguments
parser.add_argument('--patch_filter'   , '-pf', default = 0        , type=int , help = 'What patch on the sky to use for training')
parser.add_argument('--instrument'     , '-i' , default = None     , type=str , help = 'Filter or intrument used in telescope')


parser2 = argparse.ArgumentParser(description='Model Class Arguments')

parser2.add_argument('--patch_filter'   , '-pf',    default = 0            , type=int , help = 'What patch on the sky to use for training')
parser2.add_argument('--name'           , '-n' ,    default = ''           , type=str , help = 'Model name/number/whatever')
parser2.add_argument('--column_filter'  , '-cf',    default = 'Corr'       , type=str , help = 'Column filter to use')
parser2.add_argument('--target'         , '-tg',    default = 'both'       , type=str , help = 'Target(s) for model')
parser2.add_argument('--activation'     , '-a' ,    default = 'relu'       , type=str , help = 'Activation function in the model')


parser2.add_argument('--train'          , '-t' ,    action  = 'store_true' , help = 'Whether to train model or not')
parser2.add_argument('--save'           , '-s' ,    action  = 'store_true' , help = 'Whether to save model or not')
parser2.add_argument('--plot_sorted'    , '-psp',   action  = 'store_true' , help = 'Whether to plot sorted predictions or not')
parser2.add_argument('--load'           , '-l' ,    action  = 'store_false', help = 'Whether to load model or not')
