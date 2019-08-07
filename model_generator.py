import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.pyplot import figure
from matplotlib.legend_handler import HandlerLine2D
from statsmodels.tsa.seasonal import seasonal_decompose
from data_helper import *

import sys
import keras
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
from numpy import array
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib



import configparser
config = configparser.ConfigParser()
config.read('config/mylstmconfig.ini')
from data_helper import *
scaler = MinMaxScaler(feature_range=(0,1))
eco_tools_path = config['SETUP']['eco_tools_path']
sys.path.append(eco_tools_path)
from ecotools.pi_client import pi_client
pc = pi_client(root = 'readonly')

def append_variables(df):
    look_back = int(config['model']['look_back'])
    point_list = ['aiTIT4045','Future_TMY','Outside_Air_Temp_Forecast']
    start = df.index[0].strftime('%Y-%m-%d')
    end = df.index[-1].strftime('%Y-%m-%d')
    calculation = 'summary'
    interval = '1h'
    df1 = pc.get_stream_by_point(point_list, start = start, end = end, calculation = calculation, interval= interval)
    df = pd.concat([df, df1], axis = 1)
    df = fill_nan_and_stale_values(df,col_to_fill = 'aiTIT4045', cols_for_fill = ['Outside_Air_Temp_Forecast','Future_TMY'] , ffill = True)
    start = df.shape[0]
    df = df.dropna(how='any')
    print(f"Removed: {start - df.shape[0]} rows")
    df = create_standard_multivariable_df(df, shift = look_back)
    return df

def scale_data(X,y, trng_per):
    scaler = MinMaxScaler(feature_range=(0,1))
    train_idx = X[ : int(trng_per * len(X))].index
    test_idx = X[int(trng_per * len(X)):].index
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(np.array(y).reshape((-1,1)))

    X_train, X_test = np.split(X, [int(trng_per * len(X))])
    y_train, y_test = np.split(y, [int(trng_per * len(y))])

    return X_train, X_test, y_train, y_test, scaler, train_idx, test_idx

def create_model(df1, kwargs):
    point_name = kwargs['point']
    #if user wants to train the model on the residuals instead of the original data
    if kwargs['train_on_residuals']:
        decomposed = decompose_data(df1[point_name], method = kwargs['method'])
        data = {}
        #Create a dict with (each object in tuple is a pandas Series)
        # Data: (training, testing), Trend:(training, testing), Seasonality:(training, testing), Noise: (training, testing)
        val = 'Noise'
        for col in decomposed:
            data[col] = split_data(decomposed[col], split = kwargs['training_percent'])
        df = pd.concat([data[val][0], data[val][1]]).to_frame()
        df.rename(columns={val : point_name}, inplace = True)
        
    else:
        df = df1.copy()
    df.fillna(method = 'ffill', inplace = True)
    df = append_variables(df)

    y = df[point_name]
    X = df.drop(columns=point_name)

    if kwargs['model_type'] =='LSTM':
        show_every = int(config['outfiles']['show_every'])
        training_percent =float(kwargs['training_percent']) 
        validation_split = float(1.0 - training_percent)
        epochs = int(config['model']['epochs'])
        n_jobs = int(config['model']['n_jobs'])
        neurons = int(config['model']['neurons'])
        X_train, X_test, y_train, y_test, scaler , train_idx, test_idx = scale_data(X,y, training_percent) 
        # reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        train = DataFrame()
        val = DataFrame()
        np.random.seed(42)
        epochs = 10
        for i in range(n_jobs):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
            model.add(LSTM(neurons))
            model.add(Dense(1))
            model.compile(optimizer = Adam(lr = 0.001), loss = 'mean_squared_error')

            # fit model
            history = model.fit(X_train, y_train, epochs = epochs, validation_split = validation_split, shuffle = False)
            # story history
            train[str(i)] = history.history['loss']
            val[str(i)] = history.history['val_loss']
        if kwargs['show_training_plot']:
            plot_model_training(train, val, epochs, neurons, show_every = show_every)
        model = history.model
        
    return model,X_train, X_test, y_train, y_test, scaler , train_idx, test_idx

