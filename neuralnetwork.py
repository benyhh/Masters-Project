import importlib
import functions
import plotter
import arguments
import dataset
import models_v2
importlib.reload(functions); importlib.reload(plotter); importlib.reload(arguments); importlib.reload(dataset)
importlib.reload(models_v2)
import settings
importlib.reload(settings)
from settings import patches, features, dataset_params

import arguments
from functions import *
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from plotter import Plotter
import sys
from IPython import embed
from functions import random_seed
from scipy.stats import ks_2samp
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
from models_v2 import get_split_indices

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
    10: (-90,0,90,90,0),

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

    'hp_az0':          ['SunElDiff', 'DEL_TILT', 'DAZ_TILT', 'DAZ_TILTTEMP', 'TILT1X_sumdabs1', 'ACTUALAZ', 'WINDDIRECTION', 'DAZ_DISP', 'SunElDiff', 'ACTUALEL',
                        'TILT1X', 'TURBULENCE', 'ROTATIONX_sumdabs1', 'TEMP1', 'WINDDIR DIFF','date']
    }


def MSDLoss(preds, targets):
    squared_distance = torch.sub(preds, targets).pow(2).sum(dim=1)
    mean_squared_distance = squared_distance.mean()
    return mean_squared_distance

def MSE_loss1(y_true, y_pred):
    y_true = y_true.unsqueeze(1)
    return F.mse_loss(y_true, y_pred)

def MSE_sphere(y_true, y_pred):
    mse = torch.mean((y_pred - y_true)**2, dim=1)

    distance_to_center = torch.norm(y_pred, dim=1)
    distance_to_surface = torch.abs(distance_to_center - 1)
    distance_squared = distance_to_surface**2

    loss = torch.mean(mse + 10*distance_squared)
    return loss

def MSE_scaled_loss(y_true, y_pred, y_scaler):

    y_true_unscaled = y_scaler.inverse_transform(y_true.clone().detach().numpy())
    cosine_tensor = torch.cos(y_true[:,1]).unsqueeze(1)
    cosine_tensor = torch.cos(torch.from_numpy(y_true_unscaled[:,1])).unsqueeze(1)

    scaling = torch.ones(y_true.shape[0], 1)
    scaling = torch.cat((cosine_tensor, scaling), dim=1)

    #Calculate the RMS for the two vectors
    x = torch.sub(y_true, y_pred)

    x = torch.mul(x, scaling)
    x = torch.pow(x, 2)
    x = torch.sum(x, 1)
    x = torch.mean(x)

    return x


class CombinedNetworks(nn.Module):
    def __init__(self, linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function, linear_layer_activation = None):
        super(CombinedNetworks, self).__init__()
        activation_dictionary = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }
        

        self.activation_function = activation_dictionary[activation_function]        
        self.linear_layer_activation = linear_layer_activation

        layers = [nn.Linear(nonlinear_input_dim, hidden_layers[0]), self.activation_function]
        layers[0].weight.data.normal_(0, np.sqrt(2.0 / nonlinear_input_dim))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_function)
            layers[-2].weight.data.normal_(0, np.sqrt(2.0 / hidden_layers[i]))

        self.nonlinear_layers = nn.Sequential(*layers)
        

        if linear_layer_activation is not None:
            linear_layer = [nn.Linear(linear_input_dim,linear_input_dim)] 
            linear_layer[0].weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim)))
            linear_layer.append(activation_dictionary[linear_layer_activation])
            self.linear_layer = nn.Sequential(*linear_layer)


        self.final_layer = nn.Linear(linear_input_dim + hidden_layers[-1], 2)
        self.final_layer.weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim + hidden_layers[-1])))

    def forward(self, X):

        linear_input, nonlinear_input = X
        # linear_output = self.linear_layer(linear_input)
        nonlinear_output = self.nonlinear_layers(nonlinear_input)
        
        if self.linear_layer_activation is not None:
            linear_input    = self.linear_layer(linear_input)

        concatenated_output = torch.cat((linear_input, nonlinear_output), dim=1)
        final_output = self.final_layer(concatenated_output)
        
        return final_output


class CombinedNetworksConnected(nn.Module):
    def __init__(self, linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function):
        super(CombinedNetworksConnected, self).__init__()
        activation_dictionary = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }
        

        self.activation_function = activation_dictionary[activation_function]        

        # self.linear_layer = nn.Linear(linear_input_dim, linear_input_dim)
        
        layers = [nn.Linear(nonlinear_input_dim, hidden_layers[0]), self.activation_function]
        layers[0].weight.data.normal_(0, np.sqrt(2.0 / nonlinear_input_dim))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_function)
            layers[-2].weight.data.normal_(0, np.sqrt(2.0 / hidden_layers[i]))
            
        self.nonlinear_layers = nn.Sequential(*layers)
        
        final_layers = [nn.Linear(linear_input_dim + hidden_layers[-1], linear_input_dim + hidden_layers[-1]), nn.Linear(linear_input_dim + hidden_layers[-1], 2)]
        final_layers[0].weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim + hidden_layers[-1])))
        final_layers[1].weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim + hidden_layers[-1])))

        self.final_layer = nn.Sequential(*final_layers)


    def forward(self, X):

        linear_input, nonlinear_input = X
        # linear_output = self.linear_layer(linear_input)
        nonlinear_output = self.nonlinear_layers(nonlinear_input)
        
        concatenated_output = torch.cat((linear_input, nonlinear_output), dim=1)
        
        final_output = self.final_layer(concatenated_output)
        
        return final_output



class PINN(nn.Module):
    #write class for pytorch newural network
    def __init__(self, input_size, output_size, hidden_layers, dropout=0.5, activation='relu',scaler = None):
        super().__init__()

        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=dropout)
        
        # #azimuth axis misalignment EW
        # self.AW = nn.Parameter(torch.randn(1, requires_grad=True))

        # # #azimuth axis misalignment NS
        # self.AN = nn.Parameter(torch.randn(1, requires_grad=True))

        # # #LR collimation error
        # self.CA = nn.Parameter(torch.randn(1, requires_grad=True))

        # # #Az/El non-perpendicularity
        # self.NPAE = nn.Parameter(torch.randn(1, requires_grad=True))

        # self.IA = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.IE = nn.Parameter(torch.randn(1, requires_grad=True))

        # self.NRX = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.NRY = nn.Parameter(torch.randn(1, requires_grad=True))

        # self.HESE = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HECE = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HASA = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HECA2 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HSCA2 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HACA3 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HASA2 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HACA2 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HSCA  = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HESA2 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HECA3 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HSCA5 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HESA3 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HESA4 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.HESA5 = nn.Parameter(torch.randn(1, requires_grad=True))


        self.scaler = scaler

        activation_dictionary = {
            'relu': F.relu,
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU('tanh'),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }

        self.activation = activation_dictionary[activation]

    def forward(self, x):
        # # Forward pass through the network, returns the output logits
        # x_radians = self.scaler.inverse_transform(x)
        
        # COMMANDAZ = x_radians[:,0]
        # COMMANDEL = x_radians[:,1]
        # #Turn commandaz and commandel into torch tensors
        # COMMANDAZ = torch.from_numpy(COMMANDAZ).float()
        # COMMANDEL = torch.from_numpy(COMMANDEL).float()


        for each in self.hidden_layers:
            x = self.activation(each(x))
            x = self.dropout(x)
        x = self.output(x)

        # CA    = -self.CA/torch.cos(COMMANDEL)

        # AN_AZ = - self.AN * torch.sin(COMMANDAZ) * torch.tan(COMMANDEL)
        # AN_EL = - self.AN * torch.cos(COMMANDAZ)

        # AW_AZ = - self.AW * torch.cos(COMMANDAZ) * torch.tan(COMMANDEL)
        # AW_EL = self.AW * torch.sin(COMMANDAZ)

        # NPAE  = - self.NPAE * torch.tan(COMMANDEL)

        # HESE  = self.HESE * torch.sin(COMMANDEL)
        # HECE  = self.HECE * torch.cos(COMMANDEL)
        # HASA  = self.HASA * torch.sin(COMMANDAZ)
        # HECA2 = self.HECA2 * torch.cos(2*COMMANDAZ)
        # HSCA2 = self.HSCA2 * torch.sin(2*COMMANDAZ) / torch.cos(COMMANDEL)
        # HACA3 = self.HACA3 * torch.cos(3*COMMANDAZ)
        # HASA2 = self.HASA2 * torch.sin(2*COMMANDAZ)
        # HACA2 = self.HACA2 * torch.cos(2*COMMANDAZ)
        # HSCA  = self.HSCA * torch.sin(COMMANDAZ) / torch.cos(COMMANDEL)
        # HESA2 = self.HESA2 * torch.sin(2*COMMANDAZ)
        # HECA3 = self.HECA3 * torch.cos(3*COMMANDAZ)
        # HSCA5 = self.HSCA5 * torch.cos(5*COMMANDAZ) / torch.cos(COMMANDEL)
        # HESA3 = self.HESA3 * torch.sin(3*COMMANDAZ)
        # HESA4 = self.HESA4 * torch.sin(4*COMMANDAZ)
        # HESA5 = self.HESA5 * torch.sin(5*COMMANDAZ)



        #x[:,0] = x[:,0] + CA + AN_AZ + AW_AZ + NPAE #+ HASA + HSCA2 + HACA3 + HASA2 + HACA2 + HSCA + HSCA5
        #x[:,1] = x[:,1] + AN_EL + AW_EL #+ HESE + HECE + HECA2 + HESA2 + HECA3 + HESA3 + HESA4 + HESA5
        return x




class NeuralNetwork(nn.Module):
    #write class for pytorch newural network
    def __init__(self, input_size, output_size, hidden_layers, dropout=0, activation_function='relu'):
        super().__init__()
      

        activation_dictionary = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.activation_function = activation_dictionary[activation_function]


        layers = [nn.Linear(input_size, hidden_layers[0]), self.activation_function]
        layers[0].weight.data.normal_(0, np.sqrt(2. / input_size))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_function)
            layers[-2].weight.data.normal_(0, np.sqrt(2.0 / hidden_layers[i]))

        layers.append(nn.Linear(hidden_layers[-1], output_size))


        self.hidden_layers = nn.Sequential(*layers)

        

    def forward(self, x):
        # Forward pass through the network, returns the output logits

        x = self.hidden_layers(x)
    
        return x



class PrepareData():

    def __init__(self,
                df_path              = './Data/merged_features3_all.csv',
                target               = 'both',
                selected_columns_key = 'Corr_reduced2', 
                patch_key            = 0,
                n_components         = 0.99
                ) -> None:

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

        self.target         = target
        self.targets        = targets
        self.scaled         = False


        df_pointing = pd.read_csv('./Data/PointingTable.csv') # df with offsets
        df_pointing.insert(0, 'Offset', np.sqrt(df_pointing['Off_El']**2 + df_pointing['Off_Az']**2))

        self.instruments = list(df_pointing['rx'].unique())
        self.df = self.df.merge(df_pointing.loc[: , ['obs_date', 'ca', 'ie', 'rx'] + targets[target]], how = 'left', left_on='date', right_on='obs_date')
        dummies = pd.get_dummies(self.df['rx'])
        self.df = pd.concat([self.df.loc[: , self.df.columns != 'rx'], dummies], axis = 1)



        self.df = self.df.loc[ : , self.df.columns != 'obs_date']
        self.df = self.df.drop_duplicates(subset = ['date'], keep = 'first')
        
        #Removes actualaz and actualel, and replaces them with cartesian coordinate
        self.use_cartesian()

        if polluted:
            self.df.insert(0, 'polluted_az', self.df['Off_Az'])
            self.df.insert(0, 'polluted_el', self.df['Off_El'])

        self.remove_outliers()
        # self.remove_outliers(from_target=False)
        self.train_test_split_days()

        X_train, y_train = self.split_df(self.df_train, target = targets[target])
        X_test, y_test   = self.split_df(self.df_test , target = targets[target])

        X_train, X_test, y_train, y_test = self.scale_data(X_train, X_test, y_train, y_test)
        X_train, X_test                  = self.PCA(X_train, X_test, n_components)

        #Turn x and y into attributes
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test


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
            print('After PCA:',X_train.shape)
    
        return X_train, X_test

    def use_cartesian(self):
        df = self.df
        df['ACTUALAZ'] = np.deg2rad(df['ACTUALAZ'])
        df['ACTUALEL'] = np.deg2rad(df['ACTUALEL'])

        x = np.sin(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
        y = np.cos(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
        z = np.sin(df['ACTUALEL'].values)

        #Insert x,y and z, and remove ACTUALAZ and ACTUALEL from df
        df.insert(0, 'X', x)
        df.insert(0, 'Y', y)
        df.insert(0, 'Z', z)

        df = df.loc[ : , ~df.columns.isin(['ACTUALAZ', 'ACTUALEL'])]

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
        return

    def remove_outliers(self, from_target = True):
        non_val_cols = ['Hour', 'date']
        if from_target:
            factor = 2
            non_val_cols = ['Off_El', 'Off_Az'] # one or more

        else:
            factor = 1.7
            non_val_cols = list(self.df.loc[: , ~self.df.columns.isin(['Off_Az', 'Off_El', 'Hour', 'date', 'SunAboveHorizon','ie','ca','TEMP1','TEMP27','TILT1T'] + self.instruments)].columns)

        Q1 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.25)
        Q3 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.75)
        IQR = Q3 - Q1

        ## Will raise ValueError in the future
        self.df = self.df[~((self.df.loc[: , self.df.columns.isin(non_val_cols)] < (Q1 - factor * IQR)) |(self.df.loc[: , self.df.columns.isin(non_val_cols)] > (Q3 + factor * IQR))).any(axis=1)]
        
        return

    def split_df(self, df, target):
        X = df.loc[:, ~ df.columns.isin( ['date'] + target )]
        self.xcols = X.columns
        y = df.loc[:, target]
        return X, y


    def scale_data(self, X_train, X_test, y_train, y_test, scaler = 'PowerTransformer'):
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
            y = self.scaler2.inverse_transform(y)
            
        else:
            y = self.scaler2.inverse_transform(y.reshape(-1,1)).ravel()

        self.scaled = False
        
        return y


class MyDataset(Dataset):

    def __init__(self, X, y):

        #Turn x and y to numpy arrays if they are no
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values

        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyCombinedDataset(Dataset):
    def __init__(self, X_linear, X_nonlinear, y):
        
        if not isinstance(X_linear, np.ndarray):
            X_linear = X_linear.values
        if not isinstance(X_nonlinear, np.ndarray):
            X_nonlinear = X_nonlinear.values
        if not isinstance(y, np.ndarray):
            y = y.values
        
        if not torch.is_tensor(X_linear):
            self.X_linear = torch.from_numpy(X_linear)

        if not torch.is_tensor(X_nonlinear):
            self.X_nonlinear = torch.from_numpy(X_nonlinear)
        
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        
    def __len__(self):
        return self.X_linear.size(0)
    
    def __getitem__(self, idx):
        return (self.X_linear[idx], self.X_nonlinear[idx]), self.y[idx]




def train(model, train_loader, test_loader, params, PATH_MODEL='', PATH_LOSSCURVE=''):
    
    FULL_PATH_MODEL = os.path.join(PATH_MODEL, f'model_{params["name"]}.pt') 

    num_epochs = params["num_epochs"]
    learning_rate = params['learning_rate']
    best_val_measure = 1e6
    loss_func = params['loss_func']

    plotter = Plotter(name=params['name'], path = PATH_LOSSCURVE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    all_losses = []
    for e in range(num_epochs):
        batch_losses = []

        for ix, (Xb, yb) in enumerate(train_loader):
            if isinstance(Xb, list):
                _X = (Xb[0].float(), Xb[1].float())
            else:
                _X = Xb.float()

            _y = yb.float()


            #==========Forward pass===============

            preds = model(_X)
            loss = loss_func(_y, preds)     
            #==========backward pass==============

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.data)
            all_losses.append(loss.data)


        mbl = np.mean(batch_losses)

        if e % 5 == 0:
            model.eval()
            test_batch_losses = []
            for _X, _y in test_loader:

                if isinstance(_X, list):
                    _X = (_X[0].float(), _X[1].float())
                else:
                    _X = _X.float()
                
                _y = _y.float()

                #apply model
                test_preds = model(_X)
                test_loss = loss_func(_y, test_preds)
                test_batch_losses.append(test_loss.data)

            mvl = np.mean(test_batch_losses)
            print(f"Epoch [{e+1}/{num_epochs}], Batch loss: {mbl}, Val loss: {mvl}")
            if mvl < best_val_measure:
                best_val_measure = mvl
                torch.save(model.state_dict(), FULL_PATH_MODEL)

            plotter.update_withval(e, mbl, mvl, 'val')
            model.train()

        else:
            print(f"Epoch [{e+1}/{num_epochs}], Batch loss: {mbl}")
            plotter.update(e, mbl, 'train')

    model.load_state_dict(torch.load(FULL_PATH_MODEL))
    model.eval()
    return model


def plot_sorted_predictions(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED):
    print(f"Plotting sorted predictions for {params['name']}")

    FULL_PATH_SORTPRED = os.path.join(PATH_SORTPRED, f'sortpred_{params["name"]}.png')
    FULL_PATH_MODEL    = os.path.join(PATH_MODEL, f'model_{params["name"]}.pt')

    

    y_true, y_pred = [], []

    
    for _X, _y in test_loader:

        _X = Variable(_X).float()
        _y = Variable(_y).float()

        #apply model
        test_preds = pred_func(model, _X)

        y_true.append(_y.data.numpy())
        y_pred.append(test_preds.data.numpy())

    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if hasattr(ds, "y_scaler"):
        y_true = ds.rescale_y(y_true)
        y_pred = ds.rescale_y(y_pred)

    plt.clf()
    n_targets = ds.n_targets
    if n_targets > 1:
        x_res = y_true[:,0]-y_pred[:,0]
        x_res *= np.cos(y_true[:,1])
        x_res = np.rad2deg(x_res) * 3600
        
        y_res = y_true[:,1]-y_pred[:,1]
        y_res = np.rad2deg(y_res) * 3600
    
        RMS_az = np.sqrt(np.mean(x_res**2))
        RMS_el = np.sqrt(np.mean(y_res**2))
        mean_loss = np.sqrt(np.mean(np.linalg.norm(np.stack([x_res, y_res], axis = 1), axis = 1)**2))
        
        no_prediction = np.mean(np.linalg.norm(y_true, axis = 1))
        idxSorted = [y_true[:,i].argsort() for i in range(n_targets)]

        for i in range(n_targets):
            plt.plot(range(len(y_pred[:,i])), y_pred[idxSorted[i],i], label=f"Prediction {i}")
            
            plt.plot(range(len(y_pred[:,i])), y_true[idxSorted[i],i], label=f'True {i}')

    elif n_targets == 1:
        mean_loss     = np.mean(np.abs(y_true-y_pred))
        no_prediction = np.mean(np.abs(y_true))
        
        idxSorted = y_true.argsort()

        plt.plot(range(len(y_pred)), y_pred[idxSorted], label="Prediction")
        
        plt.plot(range(len(y_pred)), y_true[idxSorted], label='True')

    else:
        print('Number of targets not valid')

    plt.xlabel("Sample #")
    plt.ylabel("Offset [arcseconds]")
    print(f'Azimuth RMS: {RMS_az:.3f} arcsecs | Elevation RMS: {RMS_el:.3f}')
    print(f'RMS: {mean_loss:.3f} arcsecs')
    plt.title(f"Neural Network | ME: {mean_loss:.3f} arcsecs | {no_prediction:.3f}")
    plt.legend()
    
    plt.savefig(FULL_PATH_SORTPRED, dpi = 400)

    return mean_loss

def plot_sorted_predictions_v2(model, test_loader, y_scaler, name, PATH_SORTPRED):
    print(f"Plotting sorted predictions for {name}")

    FULL_PATH_SORTPRED = os.path.join(PATH_SORTPRED, f'sortpred_{name}.png')
    

    y_true, y_pred = [], []

    
    for _X, _y in test_loader:

        _X = Variable(_X).float()
        _y = Variable(_y).float()

        #apply model
        test_preds = model(_X)

        y_true.append(_y.data.numpy())
        y_pred.append(test_preds.data.numpy())

    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(y_pred)

    plt.clf()       
    n_targets = y_true.shape[1]
    if n_targets > 1:
        x_res = y_true[:,0]-y_pred[:,0]
        x_res *= np.cos(y_true[:,1])
        x_res  = np.rad2deg(x_res) * 3600

        y_res = y_true[:,1]-y_pred[:,1]
        y_res = np.rad2deg(y_res) * 3600

        mean_loss = np.sqrt(np.mean(np.linalg.norm(np.stack([x_res, y_res], axis = 1), axis = 1)**2))
        RMS_az = np.sqrt(np.mean(x_res**2))
        RMS_el = np.sqrt(np.mean(y_res**2))
        no_prediction = np.mean(np.linalg.norm(y_true, axis = 1))
        idxSorted = [y_true[:,i].argsort() for i in range(n_targets)]

        for i in range(n_targets):
            plt.plot(range(len(y_pred[:,i])), y_pred[idxSorted[i],i], label=f"Prediction {i}")
            
            plt.plot(range(len(y_pred[:,i])), y_true[idxSorted[i],i], label=f'True {i}')

    elif n_targets == 1:
        mean_loss     = np.mean(np.abs(y_true-y_pred))
        no_prediction = np.mean(np.abs(y_true))
        
        idxSorted = y_true.argsort()

        plt.plot(range(len(y_pred)), y_pred[idxSorted], label="Prediction")
        
        plt.plot(range(len(y_pred)), y_true[idxSorted], label='True')

    else:
        print('Number of targets not valid')

    plt.xlabel("Sample #")
    plt.ylabel("Offset [arcseconds]")
    print(f'Azimuth RMS: {RMS_az:.3f} arcsecs | Elevation RMS: {RMS_el:.3f}')
    print(f'RMS: {mean_loss:.3f} arcsecs')
    plt.title(f"Neural Network | RMS: {mean_loss:.3f} arcsecs ")
    plt.legend()
    
    plt.savefig(FULL_PATH_SORTPRED, dpi = 400)

    return mean_loss

def plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED):
    print(f"Plotting sorted predictions for {params['name']}")

    FULL_PATH_SORTPRED = os.path.join(PATH_SORTPRED, f'sortpred_{params["name"]}.png')
    FULL_PATH_MODEL    = os.path.join(PATH_MODEL, f'model_{params["name"]}.pt')

    

    y_true, y_pred = [], []

    
    for ix, (Xb, yb) in enumerate(test_loader):
        if isinstance(Xb, list):
            _X = (Xb[0].float(), Xb[1].float())
        else:
            _X = Xb.float()

        _y = yb.float()


        if isinstance(Xb, list):
            _X = (Xb[0].float(), Xb[1].float())
        else:
            _X = Xb.float()

        _y = yb.float()
        
        #apply model
        test_preds = model(_X)

        y_true.append(_y.data.numpy())
        y_pred.append(test_preds.data.numpy())

    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if hasattr(ds, "y_scaler"):
        y_true = ds.rescale_y(y_true)
        y_pred = ds.rescale_y(y_pred)

    y_true = np.rad2deg(y_true) * 3600
    y_pred = np.rad2deg(y_pred) * 3600


    plt.clf() 
    n_targets = y_true.shape[1]
    if n_targets > 1:
        x_res = y_true[:,0]-y_pred[:,0]    
        y_res = y_true[:,1]-y_pred[:,1]
    
        RMS_az = np.sqrt(np.mean(x_res**2))
        RMS_el = np.sqrt(np.mean(y_res**2))
        mean_loss = np.sqrt(np.mean(np.linalg.norm(np.stack([x_res, y_res], axis = 1), axis = 1)**2))
        
        no_prediction = np.mean(np.linalg.norm(y_true, axis = 1))
        idxSorted = [y_true[:,i].argsort() for i in range(n_targets)]

        for i, _target in zip(range(n_targets), ['Azimuth', 'Elevation']):
            plt.plot(range(len(y_pred[:,i])), y_pred[idxSorted[i],i], label=f"Predicted {_target}")
            
            plt.plot(range(len(y_pred[:,i])), y_true[idxSorted[i],i], label=f'True {_target}')

    elif n_targets == 1:
        mean_loss     = np.mean(np.abs(y_true-y_pred))
        no_prediction = np.mean(np.abs(y_true))
        
        idxSorted = y_true.argsort()

        plt.plot(range(len(y_pred)), y_pred[idxSorted], label="Prediction")
        
        plt.plot(range(len(y_pred)), y_true[idxSorted], label='True')

    else:
        print('Number of targets not valid')

    plt.xlabel("Sample #")
    plt.ylabel("Offset [arcseconds]")
    print(f'Azimuth RMS: {RMS_az:.3f} arcsecs | Elevation RMS: {RMS_el:.3f}')
    print(f'RMS: {mean_loss:.3f} arcsecs')
    plt.title(f"Neural Network | RMS: {mean_loss:.2f} arcsecs")
    plt.legend()
    
    plt.savefig(FULL_PATH_SORTPRED, dpi = 400)

    return RMS_az, RMS_el, mean_loss


def main():
    args = arguments.parser2.parse_args()

    column_filter = args.column_filter
    patch_filter  = args.patch_filter
    name          = args.name

    #Good for grid search
    PATH_SORTPRED   = f'./Results/{patch_filter}/{column_filter}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./Results/{patch_filter}/{column_filter}/LossCurve/'
    PATH_MODEL      = f'./Results/{patch_filter}/{column_filter}/Model/'

    PATH_SORTPRED   = f'./Results/SortPred/'
    PATH_LOSSCURVE  = f'./Results/LossCurve/'
    PATH_MODEL      = f'./Results/Model/'

    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL]:
        if not os.path.exists(p):
            os.makedirs(p)

    ds = PrepareData(
            selected_columns_key = args.column_filter,
            patch_key            = args.patch_filter,
            target               = args.target
            )


    train_set = MyDataset(ds.X_train, ds.y_train)
    test_set  = MyDataset(ds.X_test, ds.y_test)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=32, shuffle=True)

    input_size  = ds.X_train.size(1)
    output_size = len(ds.y_train.size())
    activation  = args.activation


    model = NeuralNetwork(input_size=input_size, output_size = output_size, hidden_layers=[400,180,250], dropout=0.3, activation = activation)


    model = train(model, train_loader, test_loader, loss_func, name, PATH_MODEL, PATH_LOSSCURVE)

    plot_sorted_predictions(model, pred_single, test_loader, ds, name, PATH_MODEL, PATH_SORTPRED)



def main_dict(params):
    args = arguments.parser2.parse_args()

    column_filter = params['column_filter']
    patch_filter  = params['patch_filter']

    #Good for grid search
    PATH_SORTPRED   = f'./Results/{patch_filter}/{column_filter}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./Results/{patch_filter}/{column_filter}/LossCurve/'
    PATH_MODEL      = f'./Results/{patch_filter}/{column_filter}/Model/'

    PATH_SORTPRED   = f'./Results/RandomSearch/SortPred/'
    PATH_LOSSCURVE  = f'./Results/RandomSearch/LossCurve/'
    PATH_MODEL      = f'./Results/RandomSearch/Model/'

    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL]:
        if not os.path.exists(p):
            os.makedirs(p)

    ds = PrepareData(
            selected_columns_key = column_filter,
            patch_key            = patch_filter,
            target               = 'both'
            )


    train_set = MyDataset(ds.X_train, ds.y_train)
    test_set  = MyDataset(ds.X_test, ds.y_test)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=params['batch_size'], shuffle=True)

    input_size  = ds.X_train.size(1)
    output_size = len(ds.y_train.size())
    activation  = params['activation']


    model = NeuralNetwork(input_size=input_size, output_size = output_size, hidden_layers=params['hidden_layers'], dropout=params['dropout'], activation = params['activation'])


    model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)

    loss = plot_sorted_predictions(model, pred_single, test_loader, ds, params['name'], PATH_MODEL, PATH_SORTPRED)

    return loss

def ensamble():
    args = arguments.parser2.parse_args()

    params ={
        'hidden_layers': [300,120,180],
        'dropout': 0.3,
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'Adam',
        'loss_func': nn.MSELoss(),
        'column_filter' : ['Corr', 'Corr_reduced', 'Corr_reduced2', 'Corr_reduced3'],
        'patch_filter'  : [0,1,7,9],
        'name': args.name
    }


    column_filter = args.column_filter
    patch_filter  = args.patch_filter
    name          = args.name

    #Good for grid search
    PATH_SORTPRED   = f'./Results/{patch_filter}/{column_filter}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./Results/{patch_filter}/{column_filter}/LossCurve/'
    PATH_MODEL      = f'./Results/{patch_filter}/{column_filter}/Model/'

    PATH_SORTPRED   = f'./Results/SortPred/'
    PATH_LOSSCURVE  = f'./Results/LossCurve/'
    PATH_MODEL      = f'./Results/Model/'

    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL]:
        if not os.path.exists(p):
            os.makedirs(p)

    ds = PrepareData(
            selected_columns_key = args.column_filter,
            patch_key            = args.patch_filter,
            target               = args.target
            )


    train_set = MyDataset(ds.X_train, ds.y_train)
    test_set  = MyDataset(ds.X_test, ds.y_test)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=params['batch_size'], shuffle=True)

    input_size  = train_set.X.size(1)
    output_size = len(train_set.y.size())
    activation  = args.activation
    num_models  = 5

    if output_size == 2:
        #loss_func = MSD_loss2
        loss_func = nn.MSELoss()
    else:
        loss_func = MSE_loss1


    models = [NeuralNetwork(input_size=input_size, output_size = output_size, hidden_layers=params['hidden_layers'], dropout=params['dropout'], activation = params['activation']) for i in range(num_models)]


    models = [train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE) for model,i_model in zip(models, range(num_models))]

    plot_sorted_predictions(models, pred_ensamble, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)
    return 


def ensamble_dict(params):
    args = arguments.parser2.parse_args()

    column_filter = args.column_filter
    patch_filter  = args.patch_filter
    name          = args.name

    #Good for grid search
    PATH_SORTPRED   = f'./Results/{patch_filter}/{column_filter}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./Results/{patch_filter}/{column_filter}/LossCurve/'
    PATH_MODEL      = f'./Results/{patch_filter}/{column_filter}/Model/'

    PATH_SORTPRED   = f'./Results/SortPred/'
    PATH_LOSSCURVE  = f'./Results/LossCurve/'
    PATH_MODEL      = f'./Results/Model/'

    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL]:
        if not os.path.exists(p):
            os.makedirs(p)

    ds = PrepareData(
            selected_columns_key = args.column_filter,
            patch_key            = args.patch_filter,
            target               = 'both'
            )


    train_set = MyDataset(ds.X_train, ds.y_train)
    test_set  = MyDataset(ds.X_test, ds.y_test)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=True)

    input_size  = ds.X_train.size(1)
    output_size = len(ds.y_train.size())
    activation  = args.activation
    num_models  = 5

    if output_size == 2:
        #loss_func = MSD_loss2
        loss_func = nn.MSELoss()
    else:
        loss_func = MSE_loss1


    models = [NeuralNetwork(input_size=input_size, output_size = output_size, hidden_layers=params['hidden_layers'], dropout=params['dropout'], activation = params['activation']) for i in range(num_models)]


    models = [train(model, train_loader, test_loader, loss_func, f'{name}_{i_model}', PATH_MODEL, PATH_LOSSCURVE) for model,i_model in zip(models, range(num_models))]

    plot_sorted_predictions(models, pred_ensamble, test_loader, ds, name, PATH_MODEL, PATH_SORTPRED)
    return 


def pred_ensamble(models, x):
    pred = [model(x) for model in models]

    return torch.mean(torch.stack(pred), 0)

def pred_single(model, x):
    return model(x)


def parameter_sampling():
    """
    Returns randomly sampled parameters for model.
    """
    # parameter_space ={
    #     'hidden_layers': randint.rvs(50,400, size=3),
    #     'dropout': uniform.rvs(0.1,0.5),
    #     'activation': ['relu', 'leaky_relu', 'tanh'],
    #     'learning_rate': uniform.rvs(0.0001,0.01),
    #     'batch_size': randint.rvs(32, 128),
    #     'optimizer': ['Adam'],
    #     'loss_func': [MSD_loss2, nn.MSELoss()],
    #     'column_filter' : ['Corr', 'Corr_reduced', 'Corr_reduced2', 'Corr_reduced3'],
    #     'patch_filter'  : [0,1,7,9]
    # }

    parameter_space ={
        'hidden_layers': randint.rvs(20, 120, size=randint.rvs(1, 3)),
        'activation': ['gelu', 'tanh', 'relu'],
        'learning_rate': uniform.rvs(0.001, 0.02),
        'batch_size': randint.rvs(32, 512),
        'loss_func': [nn.MSELoss(), MSDLoss],
    }

    params = {}

    for key in parameter_space.keys():
        if type(parameter_space[key]) is list:
            params[key] = random.choice(parameter_space[key])
        else:
            params[key] = parameter_space[key]

    return params

def RandomizedSearch(num_fits):

    #open csv file and write params headers as first line
    f = open('./Results/NNSearch/results_main.csv', 'w+')
    f.write('name,hidden_layers,dropout,activation,learning_rate,batch_size,optimizer,loss_func,loss,column_filter,patch_filter\n')

    for i in range(num_fits):
        params = parameter_sampling()
        params['name'] = f'rs{i}'
        print(f'Fitting model {i} with parameters: {params}')
        params['loss'] = main_dict(params)

        f.write(f"{params['name']},{params['hidden_layers']},{params['dropout']},{params['activation']},{params['learning_rate']},{params['batch_size']},{params['optimizer']},{params['loss_func']},{params['loss']},{params['column_filter']},{params['patch_filter']}\n")
    
        
def read_output(filename):

    # Regular expression pattern to match the model parameters
    param_pattern = re.compile(r"Fitting model .* with parameters: (.*)")

    # Regular expression pattern to match the validation loss
    val_loss_pattern = re.compile(r"Val loss: (\d+\.\d+)")

    # Dictionary to store the minimum validation loss for each model
    min_val_loss = {}

    # Read the file line by line
    with open(filename, "r") as f:
        # Initialize variables to track the current model and its parameters
        cur_model = None
        cur_params = None

        for line in f:
            # Check if this line contains the model parameters
            param_match = param_pattern.match(line)
            if param_match:
                # Extract the parameters and store them in the current model
                cur_params = eval(param_match.group(1))
                cur_model = cur_params["name"]

            # Check if this line contains the validation loss
            val_loss_match = val_loss_pattern.match(line)
            if val_loss_match:
                # Extract the validation loss and update the minimum value for this model
                val_loss = float(val_loss_match.group(1))
                if cur_model not in min_val_loss or val_loss < min_val_loss[cur_model]:
                    min_val_loss[cur_model] = val_loss

        # Print the parameters and minimum validation loss for each model
        for model, loss in min_val_loss.items():
            print(f"Model {model}:")
            print(f"  Parameters: {cur_params}")
            print(f"  Minimum validation loss: {loss}")
        
# define a function to extract model parameters and minimum val loss from a string
def extract_model_params_and_min_val_loss(model_str):
    # initialize model parameters and minimum val loss
    model_params = {}
    min_val_loss = float("inf")
    
    # split the string into lines
    lines = model_str.strip().split("\n")

    # iterate over the lines
    for line in lines:
        # if the line starts with "Fitting model"
        if line.startswith("Fitting model"):
        # extract the model parameters from the line
        # and add them to the dictionary
            model_params = line[line.index("{"):line.index("}")+1]
        # if the line contains "Val loss"
        elif "Val loss" in line:
        # extract the validation loss from the line
            val_loss = float(line.split(":")[-1].strip())
        # update the minimum validation loss
        min_val_loss = min(min_val_loss, val_loss)

    # return the model parameters and minimum val loss
    return model_params, min_val_loss

def read_model():

    with open('out1.log', 'r') as f:
        data = f.read()
        #print(data.split('Plotting sorted predictions for')[0])
        
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[0]))
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[1]))
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[2]))
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[3]))
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[4]))
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[5]))
        print(extract_model_params_and_min_val_loss(data.split('Plotting sorted predictions for')[6]))
        

from dataset import PrepareDataAnalytical_v2, PrepareDataRaw, PrepareDataRaw_v2

def main_v2():
    args = arguments.parser2.parse_args()

    params ={
        'hidden_layers': [200,140,200],
        'dropout': 0.2,
        'activation': 'tanh',
        'learning_rate': 0.01,
        'batch_size': 20000,
        'optimizer': 'Adam',
        'loss_func': nn.MSELoss(),
        'name': 'analytical_timeints',
        'num_epochs': 20,
        'use_cartesian': False,
        'use_time_filter': True,
    } 

    #Good for grid search
    PATH_SORTPRED   = f'./AnalyticalModel/SortedPrediction/'
    PATH_LOSSCURVE  = f'./AnalyticalModel/LossCurve/'
    PATH_MODEL      = f'./AnalyticalModel/Model/'

    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL]:
        if not os.path.exists(p):
            os.makedirs(p)



    ds = PrepareDataAnalytical_v2(target_key = 'both', use_cartesian=params['use_cartesian'], use_time_filter=params['use_time_filter'])
    
    print(type(ds.X_train),type(ds.X_test),type(ds.y_train),type(ds.X_train))

    train_set = MyDataset(ds.X_train, ds.y_train)
    test_set  = MyDataset(ds.X_test, ds.y_test)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)
    
    print(train_set.X.size(), train_set.y.size())
    input_size  = train_set.X.size(1)
    output_size = train_set.y.size(1)

    model = NeuralNetwork(input_size=input_size, output_size = output_size, hidden_layers=params['hidden_layers'], dropout=params['dropout'], activation = params['activation'])
    model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)

    plot_sorted_predictions(model, pred_single, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

def main_pinn():
    args = arguments.parser2.parse_args()

    params ={
        'hidden_layers': [64,64],
        'dropout': 0,
        'activation': 'gelu',
        'learning_rate': 0.01,
        'batch_size': 180,
        'optimizer': 'Adam',
        'weight_decay': 0,
        'loss_func': MSE_scaled_loss, #nn.MSELoss(),
        'name': 'optical_optimal_ts01',
        'num_epochs': 250,
        'use_cartesian': False,
        'target_key': 'both'
    } 

    dataset_params['name'] = params['name']
    
    with open(f'AnalyticalModelRaw/Parameters/{params["name"]}', 'wb') as f:
        pickle.dump(params, f)
    #Good for grid search
    PATH_SORTPRED   = f'./AnalyticalModelRaw/SortedPrediction/'
    PATH_LOSSCURVE  = f'./AnalyticalModelRaw/LossCurve/'
    PATH_MODEL      = f'./AnalyticalModelRaw/Model/'
    PATH_SAGE       = f'./AnalyticalModelRaw/Sage/'
    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL]:
        if not os.path.exists(p):
            os.makedirs(p)


    path_features = None#'./Data/processed_optical/features_optical.csv'
    ds = PrepareDataRaw(path_features = path_features, target_key = params['target_key'], use_cartesian=params['use_cartesian'], params = dataset_params)
    # ds = PrepareDataRaw_v2(target_key = 'both', use_cartesian=params['use_cartesian'], use_scaler = True)



    if hasattr(ds,'X_scaler'):
        scaler = ds.X_scaler
    else:
        scaler = None
    print(type(ds.X_train),type(ds.X_test),type(ds.y_train),type(ds.X_train))

    train_set = MyDataset(ds.X_train, ds.y_train)
    test_set  = MyDataset(ds.X_test, ds.y_test)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)
    
    print(train_set.X.size(), train_set.y.size()) 
    input_size  = train_set.X.size(1)
    output_size = len(train_set.y.size())
    if output_size ==  1:
        train_set.y = train_set.y.unsqueeze(1)
        test_set.y = test_set.y.unsqueeze(1)

    # model = NeuralNetwork(
    #                 input_size=input_size, output_size = output_size, hidden_layers=params['hidden_layers'],
    #                 dropout=params['dropout'],
    #                 activation = params['activation'],
                    # scaler = scaler)
    model = PINN(
            input_size=input_size, output_size = output_size,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            activation = params['activation'],
            scaler = scaler)


    model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
    plot_sorted_predictions(model, pred_single, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)
    plot_sorted_predictions(model, pred_single, train_loader, ds, {'name':params['name'] + '_train'}, PATH_MODEL, PATH_SORTPRED)

def test_on_pointing_scans(name = ''):
    """
    Test model trained on optical observation on pointing scans.
    """

    PATH_MODEL = f'./AnalyticalModelRaw/Model/model_{name}.pt'
    PATH_PARAMETERS = f'./AnalyticalModelRaw/Parameters/{name}'
    PATH_SCALER_X = f'./AnalyticalModelRaw/Scaler/X_scaler_{name}.pkl'
    PATH_SCALER_Y = f'./AnalyticalModelRaw/Scaler/y_scaler_{name}.pkl'


    with open(PATH_PARAMETERS, 'rb') as f:
        params = pickle.load(f)
    with open(PATH_SCALER_X, 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATH_SCALER_Y, 'rb') as f:
        y_scaler = pickle.load(f)
    
    df = pd.read_csv('./Data/scans_nflash230_unscaled.csv')
    X = df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']].values
    y = df[['REALAZ', 'REALEL']].values

    X = X_scaler.transform(X)
    y = y_scaler.transform(y)

    test_set  = MyDataset(X,y)
    print(test_set.X.size(), test_set.y.size()) 

    test_loader = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)
    
    input_size = X.shape[1]
    output_size = y.shape[1]

    if output_size ==  1:
        train_set.y = train_set.y.unsqueeze(1)
        test_set.y = test_set.y.unsqueeze(1)

    model = PINN(
        input_size=input_size, output_size = output_size,
        hidden_layers=params['hidden_layers'],
        dropout=params['dropout'],
        activation = params['activation'],
        scaler = X_scaler)
        
    model.load_state_dict(torch.load(PATH_MODEL))

    PATH_SORTPRED = './PretrainedModel/SortedPrediction/'
    new_name = name + "_" + 'on_scans'
    plot_sorted_predictions_v2(model, test_loader, y_scaler, new_name, PATH_SORTPRED)



def finetune_on_pointing_scans(name = ''):
    """
    Test model trained on optical observation on pointing scans.
    """

    PATH_MODEL = f'./AnalyticalModelRaw/Model/model_{name}.pt'
    PATH_PARAMETERS = f'./AnalyticalModelRaw/Parameters/{name}'
    PATH_SCALER_X = f'./AnalyticalModelRaw/Scaler/X_scaler_{name}.pkl'
    PATH_SCALER_Y = f'./AnalyticalModelRaw/Scaler/y_scaler_{name}.pkl'


    with open(PATH_PARAMETERS, 'rb') as f:
        params = pickle.load(f)
    with open(PATH_SCALER_X, 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATH_SCALER_Y, 'rb') as f:
        y_scaler = pickle.load(f)
    
    df = pd.read_csv('./Data/scans_nflash230_unscaled.csv')
    X = df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']].values
    y = df[['REALAZ', 'REALEL']].values

    X = X_scaler.transform(X)
    y = y_scaler.transform(y)

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.6, random_state = random_seed)
    X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size = 0.5, random_state = random_seed)
    
    train_set = MyDataset(X_train, y_train)
    val_set   = MyDataset(X_val, y_val)
    test_set  = MyDataset(X_test, y_test)

    print('Traning set')
    print(train_set.X.size(), train_set.y.size())
    print('Validation set')
    print(val_set.X.size(), val_set.y.size())
    print('Test set')
    print(test_set.X.size(), test_set.y.size()) 

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)


    input_size = X.shape[1]
    output_size = y.shape[1]

    if output_size ==  1:
        train_set.y = train_set.y.unsqueeze(1)
        test_set.y = test_set.y.unsqueeze(1)

    model = PINN(
        input_size=input_size, output_size = output_size,
        hidden_layers=params['hidden_layers'],
        dropout=params['dropout'],
        activation = params['activation'],
        scaler = X_scaler)
        
    PATH_SORTPRED = './PretrainedModel/SortedPrediction/'
    PATH_MODEL_NEW = './PretrainedModel/Model/'
    PATH_LOSSCURVE = './PretrainedModel/LossCurve/'
    
    params['name'] = name + "_" + 'finetune__ts20_20'
    params['num_epochs'] = 100

    model.load_state_dict(torch.load(PATH_MODEL))
    model = train(model, train_loader, val_loader, params, PATH_MODEL_NEW, PATH_LOSSCURVE)

    plot_sorted_predictions_v2(model, test_loader, y_scaler, params['name'], PATH_SORTPRED)

def add_model_output_to_dataset(name='', path_dataset=None,new_name=''):
    PATH_MODEL = f'./AnalyticalModelRaw/Model/model_{name}.pt'
    PATH_PARAMETERS = f'./AnalyticalModelRaw/Parameters/{name}'
    PATH_SCALER_X = f'./AnalyticalModelRaw/Scaler/X_scaler_{name}.pkl'
    PATH_SCALER_Y = f'./AnalyticalModelRaw/Scaler/y_scaler_{name}.pkl'


    with open(PATH_PARAMETERS, 'rb') as f:
        params = pickle.load(f)
    with open(PATH_SCALER_X, 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATH_SCALER_Y, 'rb') as f:
        y_scaler = pickle.load(f)
    
    df = pd.read_csv(path_dataset)
    X = df[['COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN']].values
    y = df[['REALAZ', 'REALEL']].values

    X = X_scaler.transform(X)
    y = y_scaler.transform(y)

    test_set  = MyDataset(X,y)
    print(test_set.X.size(), test_set.y.size()) 

    test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

    input_size = X.shape[1]
    output_size = y.shape[1]

    model = PINN(
        input_size=input_size, output_size = output_size,
        hidden_layers=params['hidden_layers'],
        dropout=params['dropout'],
        activation = params['activation'],
        scaler = X_scaler)
     
    model.load_state_dict(torch.load(PATH_MODEL))

    preds = []

    for i, (_X, _y) in enumerate(test_loader):
        _X = Variable(_X).float()
        _preds = model(_X)
        _preds = y_scaler.inverse_transform(_preds.detach().numpy())
        preds.append(_preds)

    preds = np.concatenate(preds, axis=0)

    df['MODELAZ'] = preds[:,0]
    df['MODELEL'] = preds[:,1]

    df['RESIDUALAZ'] = df['REALAZ'] - df['MODELAZ']
    df['RESIDUALEL'] = df['REALEL'] - df['MODELEL']


    df.to_csv(f'./Data/scans_nflash230_model_{new_name}.csv', index=False)

from dataset import PrepareDataCombined, PrepareDataNN

def NN_optical_experiment(process_number = 99):
    
    run_number = 1

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for p in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_RESULTS]:
        if not os.path.exists(p):
            os.makedirs(p)



    time_period_tests = {
        0: (pd.Timestamp('2022-05-22 06:00:00'), pd.Timestamp('2022-05-22 23:40:00')),
        1: (pd.Timestamp('2022-05-22'), pd.Timestamp('2022-07-04')),
        2: (pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-09-17 17:30:00')),
        3: (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-05-22')),
        4: (pd.Timestamp('2022-07-05'), pd.Timestamp('2022-08-12')),
    }

    prev_feats = [
                  'TILT1T_MEDIAN_1', 'ROTATIONX_MEDIAN_1', 'PRESSURE_MEDIAN_1', 'DAZ_TOTAL_MEDIAN_1', 'TEMP26_MEDIAN_1', 'TEMPERATURE_MEDIAN_1', 
                  'TEMP5_MEDIAN_1', 'TEMP3_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'TEMP27_MEDIAN_1', 'OFFSETAZ_UNSCALED_PREV', 'TEMP2_MEDIAN_1', 
                  'TEMP1_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'POSITIONY_MEDIAN_1', 'OFFSETEL_PREV', 'TEMP4_MEDIAN_1', 'TEMP28_MEDIAN_1', 'DISP_ABS3_MEDIAN_1', 
                  'DISP_ABS1_MEDIAN_1', 'ACTUALEL', 'WINDSPEED_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'OFFSETAZ_UNSCALED_PREV', 'COMMANDEL', 'OFFSETEL_PREV', 
                  'CA', 'NPAE', 'ANAZ', 'AWAZ', 'OFFSETAZ_UNSCALED_PREV', 'OFFSETEL_PREV', 'HECA2', 'HESA4', 'HSCA', 'HESA5', 'HECA3', 'HECE', 'HESA3', 
                  'HESE', 'OFFSETEL', 'HSCA5', 'OFFSETAZ_UNSCALED', 'OFFSETAZ', 'OFFSETAZ_UNSCALED_PREV', 'OFFSETEL_PREV', 'HESA2']

    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    linear_features = ['HSCA5', 'HECA2', 'HASA', 'HACA2', 'HESA4', 'HESA2', 'HESA5', 'HSCA2', 'HECA3', 'HECE', 'HESA3', 'HACA3', 'HESE', 'HSCA']
    geometrical_features = ['CA', 'ANAZ', 'ANEL', 'AWAZ', 'AWEL', 'NPAE']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    df = pd.read_csv('./Data/dataset_optical.csv')
    df['date'] = pd.to_datetime(df['date'])

    key_timeperiod = int(process_number)
    start, end = time_period_tests[key_timeperiod]

    df_tmp = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    df_tmp.dropna(axis = 1, thresh = len(df_tmp)-50, inplace = True)
    df_tmp.dropna(inplace = True)


    for i in range(100):

        num_features = np.random.randint(1, 16)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()
        
        ds = PrepareDataCombined(
            df = df_tmp,
            nonlinear_features = constant_features + nonlinear_sample,
            linear_features = geometrical_feature + harmonic_features
        )

        params = parameter_sampling()

        params['name'] = f'optical_tp{key_timeperiod}_i{i}'
        params['num_epochs'] = 300
        params['loss_func'] = nn.MSELoss()
        

        train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
        test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

        linear_input_dim    = ds.X_linear_train.shape[1]
        nonlinear_input_dim = ds.X_nonlinear_train.shape[1]
        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
        model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
        rms_az, rms_el, rms_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

        params['RMS Az'] = rms_az
        params['RMS El'] = rms_el
        params['RMS'] = rms_tot
        params['key_timeperiod'] = key_timeperiod
        params['Model #'] = i
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [linear_features]

        if i == 0:
            df_results = pd.DataFrame(params, index=[0])
        else:
            df_results = df_results.append(params, ignore_index=True)

        df_results.to_csv(f'{PATH_RESULTS}results_tp{key_timeperiod}.csv', index=False)


def NN_PCA_experiment(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2_pca999.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    n = len(df) 

    df_trainval = df.iloc[:int(n*0.85)]
    df_test     = df.iloc[int(n*0.85):]

    features = [f'PCA{i}' for i in range(14)]

    for i in range(200):    
        ds = PrepareDataNN(
            df = df_trainval.copy(),
            features = features,
            run_number = run_number,
        )

        params = parameter_sampling()

        params['name'] = f'pca999_rn{run_number}_i{i}'
        params['num_epochs'] = 300
        params['loss_func'] = nn.MSELoss()
        

        train_set = MyDataset(ds.X_train, ds.y_train)
        test_set  = MyDataset(ds.X_test, ds.y_test)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)
        
        input_size    = ds.X_train.shape[1]
        output_size   = ds.y_train.shape[1]
        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        model = NeuralNetwork(input_size = input_size, output_size = output_size, hidden_layers = hidden_layers, activation_function = activation_function)
        model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
        
        rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

        params['Model #'] = i
        params['RMS Val Az'] = rms_val_az
        params['RMS Val El'] = rms_val_el
        params['RMS Val'] = rms_val_tot

        #Convert df to  torch tensor
        X_test = df_test[features].values
        y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

        #Scale with ds.X_scaler and ds.y_scaler
        X_test = ds.X_scaler.transform(X_test)
    
        #Convert to torch tensor
        X_test = torch.from_numpy(X_test).float()

        y_pred = model(X_test)
        y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


        #Convert from radians to arcseconds, and calculate RMS
        y_pred = y_pred * 3600 * 180 / np.pi
        y_test = y_test * 3600 * 180 / np.pi

        rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
        rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

        #Calculate the rms of the magnitude of y_pred - y_test
        rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
        
        params['RMS Test Az'] = rms_test_az
        params['RMS Test El'] = rms_test_el
        params['RMS Test'] = rms_test_tot
        params['hidden_layers'] = [list(params['hidden_layers'])]

        if i == 0:
            df_results = pd.DataFrame(params, index=[0])
        else:
            df_results = df_results.append(params, ignore_index=True)

        df_results.to_csv(f'{PATH_RESULTS}pca999_rn{run_number}.csv', index=False)


def combined_separate_experiment(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    
    n = len(df) 

    df_trainval = df.iloc[:int(n*0.85)]
    df_test     = df.iloc[int(n*0.85):]


    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']



    for i in range(200):    
        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()
        
        ds = PrepareDataCombined(
            df = df_trainval.copy(),
            nonlinear_features = constant_features + nonlinear_sample,
            linear_features = geometrical_features + harmonic_features
        )

        params = parameter_sampling()

        params['name'] = f'comb_sep_nolayer_rn{run_number}_i{i}'
        params['num_epochs'] = 300
        params['loss_func'] = nn.MSELoss()
        

        train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
        test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

        linear_input_dim    = ds.X_linear_train.shape[1]
        nonlinear_input_dim = ds.X_nonlinear_train.shape[1]
        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
        model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
        
        rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

        params['Model #'] = i
        params['RMS Val Az'] = rms_val_az
        params['RMS Val El'] = rms_val_el
        params['RMS Val'] = rms_val_tot

        #Convert df to  torch tensor
        ds_test = PrepareDataCombined( 
            df = df_test,
            nonlinear_features = constant_features + nonlinear_sample,
            linear_features = geometrical_features + harmonic_features
        )

        X_linear_test = df_test[geometrical_features + harmonic_features].values
        X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
        y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

        X_linear_test = ds.lin_scaler.transform(X_linear_test)
        X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

        X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
        y_pred = model(X_test)
        y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


        #Convert from radians to arcseconds, and calculate RMS
        y_pred = y_pred * 3600 * 180 / np.pi
        y_test = y_test * 3600 * 180 / np.pi

        rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
        rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

        #Calculate the rms of the magnitude of y_pred - y_test
        rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
        
        params['RMS Test Az'] = rms_test_az
        params['RMS Test El'] = rms_test_el
        params['RMS Test'] = rms_test_tot
        params['hidden_layers'] = [list(params['hidden_layers'])]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]


        if i == 0:
            df_results = pd.DataFrame(params, index=[0])
        else:
            df_results = df_results.append(params, ignore_index=True)

        df_results.to_csv(f'{PATH_RESULTS}comb_sep_nolayer_rn{run_number}.csv', index=False)

def combined_separate_nonlinear_experiment(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    
    n = len(df) 

    df_trainval = df.iloc[:int(n*0.85)]
    df_test     = df.iloc[int(n*0.85):]


    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']



    for i in range(200):    
        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()
        
        ds = PrepareDataCombined(
            df = df_trainval.copy(),
            nonlinear_features = constant_features + nonlinear_sample,
            linear_features = geometrical_features + harmonic_features
        )

        params = parameter_sampling()

        params['name'] = f'comb_sep_nonlin_rn{run_number}_i{i}'
        params['num_epochs'] = 300
        params['loss_func'] = nn.MSELoss()
        

        train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
        test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

        linear_input_dim    = ds.X_linear_train.shape[1]
        nonlinear_input_dim = ds.X_nonlinear_train.shape[1]
        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function, linear_layer_activation=activation_function)
        model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
        
        rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

        params['Model #'] = i
        params['RMS Val Az'] = rms_val_az
        params['RMS Val El'] = rms_val_el
        params['RMS Val'] = rms_val_tot

        #Convert df to  torch tensor
        ds_test = PrepareDataCombined( 
            df = df_test,
            nonlinear_features = constant_features + nonlinear_sample,
            linear_features = geometrical_features + harmonic_features
        )

        X_linear_test = df_test[geometrical_features + harmonic_features].values
        X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
        y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

        X_linear_test = ds.lin_scaler.transform(X_linear_test)
        X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

        X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
        y_pred = model(X_test)
        y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


        #Convert from radians to arcseconds, and calculate RMS
        y_pred = y_pred * 3600 * 180 / np.pi
        y_test = y_test * 3600 * 180 / np.pi

        rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
        rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

        #Calculate the rms of the magnitude of y_pred - y_test
        rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
        
        params['RMS Test Az'] = rms_test_az
        params['RMS Test El'] = rms_test_el
        params['RMS Test'] = rms_test_tot
        params['hidden_layers'] = [list(params['hidden_layers'])]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]


        if i == 0:
            df_results = pd.DataFrame(params, index=[0])
        else:
            df_results = df_results.append(params, ignore_index=True)

        df_results.to_csv(f'{PATH_RESULTS}comb_sep_nonlin_rn{run_number}.csv', index=False)

def combined_connected_experiment(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    
    n = len(df) 

    df_trainval = df.iloc[:int(n*0.85)]
    df_test     = df.iloc[int(n*0.85):]


    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    for i in range(200):    
        num_features = np.random.randint(1, 16)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()
        
        ds = PrepareDataCombined(
            df = df_trainval.copy(),
            nonlinear_features = constant_features + nonlinear_sample,
            linear_features = geometrical_features + harmonic_features
        )

        params = parameter_sampling()

        params['name'] = f'comb_conn_rn{run_number}_i{i}'
        params['num_epochs'] = 300
        params['loss_func'] = nn.MSELoss()
        

        train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
        test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

        linear_input_dim    = ds.X_linear_train.shape[1]
        nonlinear_input_dim = ds.X_nonlinear_train.shape[1]
        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        model = CombinedNetworksConnected(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
        model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
        
        rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

        params['Model #'] = i
        params['RMS Val Az'] = rms_val_az
        params['RMS Val El'] = rms_val_el
        params['RMS Val'] = rms_val_tot

        X_linear_test = df_test[geometrical_features + harmonic_features].values
        X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
        y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values
        #Scale with ds.X_scaler and ds.y_scaler
        X_linear_test = ds.lin_scaler.transform(X_linear_test)
        X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

        X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
        y_pred = model(X_test)
        y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())

        #Convert from radians to arcseconds, and calculate RMS
        y_pred = y_pred * 3600 * 180 / np.pi
        y_test = y_test * 3600 * 180 / np.pi

        rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
        rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

        #Calculate the rms of the magnitude of y_pred - y_test
        rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
        
        params['RMS Test Az'] = rms_test_az
        params['RMS Test El'] = rms_test_el
        params['RMS Test'] = rms_test_tot
        params['hidden_layers'] = [list(params['hidden_layers'])]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]

        if i == 0:
            df_results = pd.DataFrame(params, index=[0])
        else:
            df_results = df_results.append(params, ignore_index=True)

        df_results.to_csv(f'{PATH_RESULTS}comb_conn_rn{run_number}.csv', index=False)


def visualize_torch_model():

    run_number = 99
    i = 99
    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    
    n = len(df) 

    df_trainval = df.iloc[:int(n*0.85)]
    df_test     = df.iloc[int(n*0.85):]


    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    num_features = np.random.randint(1, 16)
    #randomly select num_features from nonlinear_features
    nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()
    
    ds = PrepareDataCombined(
        df = df_trainval,
        nonlinear_features = constant_features + nonlinear_sample,
        linear_features = geometrical_features + harmonic_features
    )

    params = parameter_sampling()

    params['name'] = f'comb_conn_rn{run_number}_i{i}'
    params['num_epochs'] = 300
    params['loss_func'] = nn.MSELoss()
    

    train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
    test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

    linear_input_dim    = ds.X_linear_train.shape[1]
    nonlinear_input_dim = ds.X_nonlinear_train.shape[1]
    hidden_layers       = params['hidden_layers']
    activation_function = params['activation']

    # model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
    model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function, linear_layer_activation=activation_function)
    # model = CombinedNetworksConnected(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)


    from torchviz import make_dot

    batch = next(iter(train_loader))

    y = model([batch[0][0].float(), batch[0][1].float()])

    dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.render('CombinedNetwork_nonlinear', format='png')

    return
"""
    import shap
    from models_v2 import SAGE

    FULL_PATH_MODEL    = os.path.join(PATH_MODEL, f'model_{params["name"]}.pt')
    model.load_state_dict(torch.load(FULL_PATH_MODEL))
    SAGE(ds, model)

    #Signel sample from test loader
    X_dataset = test_set.X

    # create a SHAP explainer for your model
    explainer = shap.Explainer(model,X_dataset)
    print(explainer)
    # create a batch of input data to explain

    # calculate SHAP values for the input data
    shap_values = explainer(X_dataset)

    feature_importances = torch.mean(torch.abs(shap_values.values), axis=0)
    print(feature_importances)
    # create a summary plot of the SHAP values
    shap.summary_plot(shap_values, x, plot_type="scatter")

    # show the plot
    plt.savefig(PATH_SAGE, f'shap_{params["name"]}.png', dpi=300)

"""

def SAGE_torch(dataset, model, test_set = None, path_save='SAGE'):
    print('Using sage in models_v2')
    if test_set is not None:
        X_test, y_test = test_set.X, test_set.y
    else:
        X_train, X_test, y_train, y_test = dataset.get_data()
    
    feature_names = dataset.feature_names
    print(X_test.shape, y_test.shape)
    imputer = sage.MarginalImputer(lambda x: pytorch_model_wrapper(model, x), X_test)
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_testues = estimator(X_test, y_test)

    sage_testues.plot(feature_names=feature_names)
    plt.tight_layout()

    plt.savefig(path_save, dpi = 300)


def NN_experiment_CV(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)


    feature_list = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1',
                        'HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5',
                        'CA', 'NPAE']

    constant_features = ['COMMANDAZ', 'COMMANDEL']

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, len(feature_list))
        #randomly select num_features from nonlinear_features
        feature_sample = np.random.choice(feature_list, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['features'] = [feature_sample]

        for j in range(n_folds-1, n_folds):

            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 

            ds = PrepareDataNN(
                df = df_trainval.copy(),
                features = constant_features + feature_sample,
                run_number = run_number,
            )


            params['name'] = f'regular_rn{run_number}_i{i}_fold{j}'
            

            train_set = MyDataset(ds.X_train, ds.y_train)
            test_set  = MyDataset(ds.X_test, ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)
            
            input_size    = ds.X_train.shape[1]
            output_size   = ds.y_train.shape[1]

            model = NeuralNetwork(input_size = input_size, output_size = output_size, hidden_layers = hidden_layers, activation_function = activation_function)

            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            #Convert df to  torch tensor
            X_test = df_test[constant_features + feature_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

            #Scale with ds.X_scaler and ds.y_scaler
            X_test = ds.X_scaler.transform(X_test)
        
            #Convert to torch tensor
            X_test = torch.from_numpy(X_test).float()

            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}regular_rn{run_number}.csv', index=False)

def combined_separate_experiment_CV(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    



    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]
        
        for j in range(n_folds-1, n_folds):

            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 

            ds = PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )

            params['name'] = f'comb_sep_nolayer_rn{run_number}_i{i}_fold{j}'

            train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
            test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

            linear_input_dim    = ds.X_linear_train.shape[1]
            nonlinear_input_dim = ds.X_nonlinear_train.shape[1]

            model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            #Convert df to  torch tensor
            ds_test = PrepareDataCombined( 
                df = df_test,
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )

            X_linear_test = df_test[geometrical_features + harmonic_features].values
            X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

            X_linear_test = ds.lin_scaler.transform(X_linear_test)
            X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

            X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}comb_sep_nolayer_rn{run_number}.csv', index=False)


def combined_separate_nonlinear_experiment_CV(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    

    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    n = len(df)
    n_folds = 6
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]
     

        for j in range(n_folds):


            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 

            #print min and max index of test and tainval


            ds = PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )


            params['name'] = f'comb_sep_nonlin_rn{run_number}_i{i}_fold{j}'

            train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
            test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

            linear_input_dim    = ds.X_linear_train.shape[1]
            nonlinear_input_dim = ds.X_nonlinear_train.shape[1]

            model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function, linear_layer_activation=activation_function)
            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            #Convert df to  torch tensor
            ds_test = PrepareDataCombined( 
                df = df_test,
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )

            X_linear_test = df_test[geometrical_features + harmonic_features].values
            X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

            X_linear_test = ds.lin_scaler.transform(X_linear_test)
            X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

            X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}comb_sep_nonlin_rn{run_number}.csv', index=False)



def combined_connected_experiment_CV(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    


    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]
        
        for j in range(n_folds):


            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 
            
            ds = PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )


            params['name'] = f'comb_conn_rn{run_number}_i{i}_fold{j}'
            

            train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
            test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

            linear_input_dim    = ds.X_linear_train.shape[1]
            nonlinear_input_dim = ds.X_nonlinear_train.shape[1]

            model = CombinedNetworksConnected(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            X_linear_test = df_test[geometrical_features + harmonic_features].values
            X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values
            #Scale with ds.X_scaler and ds.y_scaler
            X_linear_test = ds.lin_scaler.transform(X_linear_test)
            X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

            X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())

            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}comb_conn_rn{run_number}.csv', index=False)





def statistics_for_raw_dataset():

    path = 'SimilarityStatistics/'

    if not os.path.exists(path):
        os.makedirs(path)

    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    relevant_features = nonlinear_features + harmonic_features + geometrical_features + constant_features
    
    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)

    num_bins = 20

    for j in range(n_folds):
        df_test = df.iloc[split_indices[j] : split_indices[j+1]]
        df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]])
        df_trainval = df_trainval[relevant_features]
        
        df_train, df_val = train_test_split(df_trainval.copy(), test_size=0.25, random_state=42)
        
        df_test = df_test[relevant_features]
        df_trainval = df_trainval[relevant_features]

        ds = PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_features,
                linear_features = geometrical_features + harmonic_features
            )

        print(f'Fold {j}')
        # compare distributions using KS test
        results = []
        for col in df_test.columns:

            min_val = df[col].min()
            max_val = df[col].max()
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            statistic, pvalue = ks_2samp(df_test[col], df_trainval[col])
            results.append({'feature': col, 'statistic': statistic, 'pvalue': pvalue})

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.hist(df_train[col], density=True, bins = bin_edges, alpha=0.5, label='train')
            ax.hist(df_val[col], density=True, bins = bin_edges, alpha=0.5, label='val')
            ax.hist(df_test[col], density=True, bins = bin_edges, alpha=0.5, label='test')

            ax.legend()
            ax.set_title(f'{col} distribution comparison')

            plt.savefig(f'{path}{col}_fold{j}_distribution_comparison.png', dpi=300)

        # sort results by p-value
        results_sorted = sorted(results, key=lambda x: x['pvalue'])

        # print results
        for result in results_sorted:
            print(f"{result['feature']} - KS statistic: {result['statistic']:.4f}, p-value: {result['pvalue']:.4f}")


def custom_format(x):
    if abs(x) < 1 or abs(x) >= 1e5:
        return '{:.1e}'.format(x)
    else:
        return '{:.2f}'.format(x)

def analyze_results_raw_model():

    df_reg = pd.read_csv('./FinalResultsOptical/Run5/regular_rn5.csv')
    df_sep1 = pd.read_csv('./FinalResultsOptical/Run5/comb_sep_nolayer_rn5.csv')
    df_sep2 = pd.read_csv('./FinalResultsOptical/Run5/comb_sep_nonlin_rn5.csv')
    df_sep3 = pd.read_csv('./FinalResultsOptical/Run5/comb_conn_rn5.csv')

    df_reg_last = pd.read_csv('./FinalResultsOptical/Run6/regular_rn6.csv')
    df_sep1_last = pd.read_csv('./FinalResultsOptical/Run6/comb_sep_nolayer_rn6.csv')


    df_reg = pd.concat([df_reg, df_reg_last])
    df_sep1 = pd.concat([df_sep1, df_sep1_last])

    df_reg.sort_values(by=['Model #', 'fold'], inplace=True)
    df_sep1.sort_values(by=['Model #', 'fold'], inplace=True)

    #Drop Model # == 0
    df_reg = df_reg[df_reg['Model #'] != 0]
    df_sep1 = df_sep1[df_sep1['Model #'] != 0]
    df_sep2 = df_sep2[df_sep2['Model #'] != 0]
    df_sep3 = df_sep3[df_sep3['Model #'] != 0]

    df_reg.to_csv('./FinalResultsOptical/Run5/regular_rn5_concat.csv', index=False)
    df_sep1.to_csv('./FinalResultsOptical/Run5/comb_sep_nolayer_rn5_concat.csv', index=False)
    

    dfs = [(df_reg, 'reg'), (df_sep1, 'sep1'), (df_sep2, 'sep2'), (df_sep3, 'sep3')]

    dfs_to_concat = []

    for df, name in dfs:
    
        df.sort_values(by='RMS Test', ascending=True, inplace=True)
        #Group by model # and calculate the the test rms mean and std over all 6 folds. All rows in these groups have one activation, add column with that too
        df_grouped = df.groupby('Model #').agg({'RMS Test': ['mean', 'std'], 'activation': 'first', 'hidden_layers': 'first', 'learning_rate': 'first', 'batch_size': 'first', 'loss_func': 'first', 'Model #': 'first'})
        #Sort by RMS Val
        df_grouped.sort_values(by=('RMS Test', 'mean'), ascending=True, inplace=True)
        print(df_grouped.iloc[:5,:7])

        best_model = df_grouped.iloc[0,7]
        df_best = df_grouped.iloc[0,:]
        df_best = df_best.droplevel(1)
        if name == 'reg':
            print(df.loc[df['Model #'] == best_model, 'features'].values[0])
            # df_best['Features'] = df.loc[df['Model #'] == best_model, 'features'].values[0]
        
        else:
            print(df.loc[df['Model #'] == best_model, 'linear_features'].values[0] + df.loc[df['Model #'] == best_model, 'nonlinear_features'].values[0])
            # df_best['Features'] = df.loc[df['Model #'] == best_model, 'linear_features'].values[0] + df.loc[df['Model #'] == best_model, 'nonlinear_features'].values[0]
        
        dfs_to_concat.append(df_best)

        # df_grouped.iloc[:,[0,1,4]] = df_grouped.iloc[:,[0,1,4]].applymap(custom_format)
        # df_grouped.iloc[:5,[2,3,4,5,6,0,1]].to_latex(f'FinalResultsOptical/Run5/results_{name}.tex', index = False)#, float_format = "%.2f")

    df_concat = pd.concat(dfs_to_concat, axis=1).T

    embed()

    df_concat.to_latex(f'FinalResultsOptical/Run5/results_best_hp_concat.tex', index = False)
    

def analyze_results_raw_model_per_fold():

    df_reg = pd.read_csv('./FinalResultsOptical/Run5/regular_rn5.csv')
    df_sep1 = pd.read_csv('./FinalResultsOptical/Run5/comb_sep_nolayer_rn5.csv')
    df_sep2 = pd.read_csv('./FinalResultsOptical/Run5/comb_sep_nonlin_rn5.csv')
    df_sep3 = pd.read_csv('./FinalResultsOptical/Run5/comb_conn_rn5.csv')

    df_reg_last = pd.read_csv('./FinalResultsOptical/Run6/regular_rn6.csv')
    df_sep1_last = pd.read_csv('./FinalResultsOptical/Run6/comb_sep_nolayer_rn6.csv')


    df_reg = pd.concat([df_reg, df_reg_last])
    df_sep1 = pd.concat([df_sep1, df_sep1_last])

    df_reg.sort_values(by=['Model #', 'fold'], inplace=True)
    df_sep1.sort_values(by=['Model #', 'fold'], inplace=True)

    #Drop Model # == 0
    df_reg = df_reg[df_reg['Model #'] != 0]
    df_sep1 = df_sep1[df_sep1['Model #'] != 0]
    df_sep2 = df_sep2[df_sep2['Model #'] != 0]
    df_sep3 = df_sep3[df_sep3['Model #'] != 0]

    df_reg.to_csv('./FinalResultsOptical/Run5/regular_rn5_concat.csv', index=False)
    df_sep1.to_csv('./FinalResultsOptical/Run5/comb_sep_nolayer_rn5_concat.csv', index=False)
    

    dfs = [(df_reg, 'reg'), (df_sep1, 'sep1'), (df_sep2, 'sep2'), (df_sep3, 'sep3')]

    dfs_to_concat = []
    for df, name in dfs:
        
        #group by model # and find the model with the lowest mean RMS Test. Filter df on that model and print the results
        df_grouped = df.groupby('Model #').agg({'RMS Test': ['mean', 'std'], 'activation': 'first', 'hidden_layers': 'first', 'learning_rate': 'first', 'batch_size': 'first', 'loss_func': 'first', 'Model #': 'first'})
        df_grouped.sort_values(by=('RMS Test', 'mean'), ascending=True, inplace=True)
        
        best_model = df_grouped.iloc[0,-1]
        df_best = df[df['Model #'] == best_model]
        print(df_best.loc[: , ['fold', 'RMS Test']])

        df_best = df_best['RMS Test'].reset_index(drop=True)
        dfs_to_concat.append(df_best)
    
    df_concat = pd.concat(dfs_to_concat, axis=1).T
    df_concat['Mean'] = df_concat.mean(axis=1)
    df_concat['STD'] = df_concat.std(axis=1)
    df_concat.to_latex(f'FinalResultsOptical/Run5/results_folds_concat.tex', index = False, float_format = "%.2f")

if __name__ == "__main__":
    #ensamble()
    #read_output('out1.log')
    # visualize_torch_model()
    # NN_PCA_experiment(run_number = 2)
    # combined_separate_experiment(run_number = 3) #beehive 19 out0 and no layer beehive 22 out 3
    # combined_separate_nonlinear_experiment(run_number = 3) #beehive 20 out1
    # combined_connected_experiment(run_number = 4) #beehive 21 out2, new one with correct class Combined Connected beehvie22 out5
    """
    nohup python -u neuralnetwork.py > out5.log 2> error5.log &
    nohup python -u neuralnetwork.py > out6.log 2> error6.log &
    nohup python -u neuralnetwork.py > out7.log 2> error7.log &
    nohup python -u neuralnetwork.py 1 > out0.log 2> error0.log &
    nohup python -u neuralnetwork.py 2 > out9.log 2> error9.log &



    nohup python -u neuralnetwork.py > out0.log 2> error0.log &
    nohup python -u neuralnetwork.py 1 > out1.log 2> error1.log &
    nohup python -u neuralnetwork.py 2 > out2.log 2> error2.log &
    nohup python -u neuralnetwork.py 3 > out3.log 2> error3.log &
    nohup python -u neuralnetwork.py 4 > out4.log 2> error4.log & 
    """
    

    print(sys.argv)
    if int(sys.argv[1]) == 1:
        NN_experiment_CV(run_number = 99)
    
    if int(sys.argv[1]) == 2:
        combined_separate_experiment_CV(run_number = 99)
    
    if int(sys.argv[1]) == 3:
        combined_separate_nonlinear_experiment_CV(run_number = 99)
    
    if int(sys.argv[1]) == 4:
        combined_connected_experiment_CV(run_number = 99)
    
    analyze_results_raw_model()
    # statistics_for_raw_dataset()

    #RandomizedSearch(1000)
    # test_on_pointing_scans('optical_optimal_ts001')
    # test_on_pointing_scans('optical_optimal_ts01')
    # test_on_pointing_scans('optical_optimal_ts03')
    
    # finetune_on_pointing_scans('optical_optimal_ts001')
    # finetune_on_pointing_scans('optical_optimal_ts01')
    # finetune_on_pointing_scans('optical_optimal_ts03')
    
    # add_model_output_to_dataset(name = 'optical_optimal_ts01', path_dataset = './Data/scans_nflash230_unscaled_all.csv', new_name = 'all_ts01')
    # add_model_output_to_dataset(name = 'optical_optimal_ts03', path_dataset = './Data/scans_nflash230_unscaled_all.csv', new_name = 'all_ts03')
    # add_model_output_to_dataset(name = 'optical_optimal_ts01', path_dataset = './Data/scans_nflash230_unscaled.csv', new_name = 'clean_ts01')
    # add_model_output_to_dataset(name = 'optical_optimal_ts03', path_dataset = './Data/scans_nflash230_unscaled.csv', new_name = 'clean_ts03')

    
    pass

    # Got the 4 arcitechtures from report running. 6- fold CV.


"""
ds = PrepareData(X, y=y, scale_X=True)

train_set = DataLoader(ds, batch_size=batch_size,
                       sampler=SubsetRandomSampler(train))
test_set = DataLoader(ds, batch_size=batch_size,
                      sampler=SubsetRandomSampler(test))
                      

1. Some sort of search for architecture
2. Find out why validation is lower than training
3. Routine for running multiple processes with different name


Hei,

Jeg har tenkt litt på hvilke tester som kan gi gode resultater for rådataen.
Vi har jo ikke noe å sammenligne med, så det blir jo ikke fungere


"""