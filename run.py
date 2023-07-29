import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
random_seed = 12
pt = pd.read_csv("PointingTable.csv")
ms = pd.read_csv("meteoscope.csv")




def PSD(y_true,y_pred,n):
    """
    Calculating the PSD (population standard deviation)
    o: number of observations
    n: number of fitted parameters in the model
    rho: the RMS of the prediction
    """
    o = len(y_true)
    rho = np.sqrt(mean_squared_error(y_true, y_pred)) 

    return np.sqrt(rho**2*o/(o-n))

def H(k,f,v):
    """
    Harmonic variable
    Parameters
    k: constant found using linear regression
    f: harmonic function, either sin or cos
    v: variable inserted into function, usually elevation or azimuth
    """
    return k*f(v)


def dE(A,E):
    """
    Calculating adjustment in elevation
    """
    return

def S2A(S,E):
    """
    Converts left-right to azimuth
    """
    return S/np.cos(E)


Az = np.radians(np.array(pt["Az"]))
El = np.radians(np.array(pt["El"]))

#Dictionary that holds the model.
modelParams = {
    "dE":{    
        "HESE": lambda A,E: np.sin(E),
        "HECE": lambda A,E: np.cos(E),
        "HECA2": lambda A,E: np.cos(2*A),
        "HESA2": lambda A,E: np.sin(2*A),
        "HECA3": lambda A,E: np.cos(3*A),
        "HESA3": lambda A,E: np.sin(3*A),
        "HESA4": lambda A,E: np.sin(4*A),
        "HESA5": lambda A,E: np.sin(5*A),
        "IE": lambda A,E: np.ones(len(A))
    },
    "dA":{
        "NPAE": lambda A,E: np.tan(E),
        "HASA": lambda A,E: np.sin(A),
        "HACA3": lambda A,E: np.cos(3*A),
        "HASA2": lambda A,E: np.sin(2*A),
        "HACA2": lambda A,E: np.cos(2*A),
        "CA": lambda A,E: 1/np.cos(E),
        "IA": lambda A,E: np.ones(len(A))
    },
    "dS":{
        "HSCA": lambda A,E: np.cos(A),
        "HSCA2": lambda A,E: np.cos(2*A),
        "HSCA5": lambda A,E: np.cos(5*A) 
    }
}


y_Az = np.radians(np.array(pt["Az"] + pt["Off_Az"]))
y_El = np.radians(np.array(pt["El"] + pt["Off_El"]))
y = np.array([y_Az, y_El])

X_El = np.array([modelParams["dE"][k](Az,El) for k in sorted(modelParams["dE"].keys())]).T
X_Az = np.array([modelParams["dA"][k](Az,El) for k in sorted(modelParams["dA"].keys())]).T
X_lr = np.array([modelParams["dS"][k](Az,El) for k in sorted(modelParams["dS"].keys())]).T

X_El_train, X_El_test, y_El_train, y_El_test = train_test_split(X_El, y_El, random_state=random_seed)


