#General Imports used across both Models
import sys
import os
import math
import warnings
#import keras
import matplotlib
import configparser
import pandas as pd
import numpy as np
from keras.optimizers import *
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from matplotlib.legend_handler import HandlerLine2D
from statsmodels.tsa.seasonal import seasonal_decompose

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

#General imports created in-house used across all script
from .data_helper import *
cwd = os.getcwd()
config_path = cwd + '/src/config/lstmconfig.ini'
config = configparser.ConfigParser()
something = config.read(config_path)
eco_tools_path = config['SETUP']['eco_tools_path']
sys.path.append(eco_tools_path)
from ecotools.pi_client import pi_client
pc = pi_client(root = 'readonly')

# LSTM Specific Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
#es = EarlyStopping(monitor='val_loss', patience = 50, mode = 'min', verbose=1)
#mc = ModelCheckpoint('best_model.h5', monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
#callbacks_list = [es, mc]

# Random Forest Imports
from sklearn.ensemble import RandomForestRegressor

#scaler = MinMaxScaler(feature_range=(0,1))


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
    df = create_standard_multivariable_df(df, shift = look_back, dropna = False)
    df.fillna(method = 'ffill', inplace = True)
    print(f"Removed: {start - df.shape[0]} rows")
    return df

def model_split_data(X,y, trng_per, scale = True):
    
    
    train_idx = X[ : int(trng_per * len(X))].index
    test_idx = X[int(trng_per * len(X)):].index
    if scale:
        scaler = MinMaxScaler(feature_range=(0,1))
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(np.array(y).reshape((-1,1)))

    X_train, X_test = np.split(X, [int(trng_per * len(X))])
    y_train, y_test = np.split(y, [int(trng_per * len(y))])
    if scale:
        return X_train, X_test, y_train, y_test, scaler, train_idx, test_idx
    else:
        return X_train, X_test, y_train, y_test, train_idx, test_idx

def generate_actual_vs_model_df(model, X, y, scaler, trend, seasonal, index, train_on_residuals = True, use_scaler = True):
    prediction = model.predict(X)
    if use_scaler:
        prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape((-1,))
        actual = scaler.inverse_transform(y).reshape((-1,))
    else:
        prediction = np.reshape(prediction,(-1,))
        actual = np.reshape(y, (-1,))
    if train_on_residuals:    
        prediction = prediction + trend + seasonal
        actual = actual + trend + seasonal
    r2 = r2_score(actual, prediction)
    rmse = math.sqrt(mean_squared_error(actual, prediction))
    mae = np.median(actual - prediction)
    return pd.DataFrame({"Actual" : actual, "Modeled" : prediction}, index = index), r2, rmse, mae

def create_model(df1, kwargs):
    point_name = kwargs['point']
    train_on_residuals = kwargs['train_on_residuals']
    model_type = kwargs['model_type']
    if model_type is not None:
        #if user wants to train the model on the residuals instead of the original data
        if train_on_residuals:
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
        df.fillna(method = 'bfill', inplace = True)
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
            X_train, X_test, y_train, y_test, scaler , train_idx, test_idx = model_split_data(X,y, training_percent) 
            # reshape input to be [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_train = np.nan_to_num(X_train)
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            train = pd.DataFrame()
            val = pd.DataFrame()
            np.random.seed(42)
            
            for i in range(n_jobs):
                model = Sequential()
                model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
                model.add(LSTM(neurons))
                model.add(Dense(1))
                model.compile(optimizer = Adam(lr = 0.001), loss = 'mean_squared_error')

                # fit model
                history = model.fit(X_train, y_train, epochs = epochs, validation_split = validation_split, shuffle = False)
                # store history
                train[str(i)] = history.history['loss']
                val[str(i)] = history.history['val_loss']
            if kwargs['show_training_plot']:
                plot_model_training(train, val, epochs, neurons, show_every = show_every)
            model = history.model

            if train_on_residuals:
                trend_training = data['Trend'][0].to_numpy()
                trend_test = data['Trend'][1].to_numpy()
                seasonal_training = data['Seasonality'][0].to_numpy()
                seasonal_test = data['Seasonality'][1].to_numpy()
                train_df, train_r2, train_rmse, train_mae = generate_actual_vs_model_df(model, 
                        X_train, y_train, scaler, 
                        trend = trend_training, seasonal = seasonal_training, index = train_idx)
                train_r2 = round(train_r2, 3)
                train_rmse = round(train_rmse, 3)
                train_mae = round(train_mae, 3)
                test_df, test_r2, test_rmse, test_mae = generate_actual_vs_model_df(model, 
                X_test, y_test, scaler, 
                trend = trend_test, seasonal = seasonal_test, index = test_idx)
            else:
                train_df, train_r2, train_rmse, train_mae = generate_actual_vs_model_df(model, 
                        X_train, y_train, scaler, 
                        trend = None, seasonal = None, index = train_idx, train_on_residuals = False)
                
                train_r2 = round(train_r2, 3)
                train_rmse = round(train_rmse, 3)
                train_mae = round(train_mae, 3)
                test_df, test_r2, test_rmse, test_mae = generate_actual_vs_model_df(model, 
                X_test, y_test, scaler, 
                trend = None, seasonal = None, index = test_idx, train_on_residuals = False)




            
            test_r2 = round(test_r2, 3)
            test_rmse = round(test_rmse, 3)
            test_mae = round(test_mae, 3) 
            if kwargs['show_results_plots']:
                plt.style.use('fivethirtyeight')
                figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')

                ax = plt.subplot()
            

                ax.plot(train_df.Actual, label = 'Training', linewidth = 1, color = 'green')
                ax.plot(test_df.Actual, label = 'Test', linewidth = 1, color = 'black')
                plt.legend(prop={'size': 14})
                plt.title('Training and Testing Data', fontsize = 14)
                plt.show()

                train_title = f"Training: Actual vs. Modeled \n R2: {train_r2}  RMSE: {train_rmse} MAE: {train_mae}"
                args = {'linewidth': 1, 'include_title': False, 'title': train_title, 
                    'include_hline': False,
                    'include_vline': False,}
                plot_data(train_df, args)
                plt.title(train_title, fontsize = 16)
                plt.ylabel(point_name)
                plt.legend(prop={'size': 14})
                plt.show()

                test_title = f"Test: Actual vs. Modeled \n R2: {test_r2}  RMSE: {test_rmse} MAE: {test_mae}"
                args = {'linewidth': 1, 'include_title': False, 'title': test_title, 
                    'include_hline': False,
                    'include_vline': False,}
                plot_data(test_df, args)
                plt.title(test_title, fontsize = 16)  
                plt.ylabel(point_name)
                plt.legend(prop={'size': 14})
                plt.show()
            else:
                print(f"  Training R2: {train_r2}")
                print(f"   Testing R2: {test_r2} \n")
                print(f"Training RMSE: {train_rmse}")
                print(f" Testing RMSE: {test_rmse} \n")
                print(f" Training MAE: {train_mae}")
                print(f"  Testing MAE: {test_mae} \n")
        else:
            n_estimators = int(config['model']['n_estimators'])
            training_percent =float(kwargs['training_percent']) 
            max_depth = int(config['model']['max_depth'])
            random_state = int(config['model']['random_state'])
            X_train, X_test, y_train, y_test,  train_idx, test_idx = model_split_data(X,y, training_percent, scale = False)
            rforest = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators, random_state= random_state,verbose = 1, n_jobs = -1)
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
            X_train.fillna(method = 'ffill', inplace = True)
            X_train.fillna(method ='bfill', inplace = True)
            y_train.fillna(method = 'ffill', inplace = True)
            y_train.fillna(method ='bfill', inplace = True)
            
            
            model = rforest.fit(X_train.fillna(method='ffill'), y_train.fillna(method='ffill'))
            
            if train_on_residuals:
                trend_training = data['Trend'][0].to_numpy()
                trend_test = data['Trend'][1].to_numpy()
                seasonal_training = data['Seasonality'][0].to_numpy()
                seasonal_test = data['Seasonality'][1].to_numpy()
                train_df, train_r2, train_rmse, train_mae = generate_actual_vs_model_df(model, 
                        X_train, y_train, scaler = None, 
                        trend = trend_training, seasonal = seasonal_training, index = train_idx, use_scaler = False)
                train_r2 = round(train_r2, 3)
                train_rmse = round(train_rmse, 3)
                train_mae = round(train_mae, 3)


                test_df, test_r2, test_rmse, test_mae = generate_actual_vs_model_df(model, 
                                X_test, y_test, scaler = None, 
                            trend = trend_test, seasonal = seasonal_test, index = test_idx, use_scaler = False)

            else:
                train_df, train_r2, train_rmse, train_mae = generate_actual_vs_model_df(model, 
                        X_train, y_train, scaler = None, 
                        trend = None, seasonal = None, index = train_idx, train_on_residuals = False, use_scaler = False)
                train_r2 = round(train_r2, 3)
                train_rmse = round(train_rmse, 3)
                train_mae = round(train_mae, 3)
                test_df, test_r2, test_rmse, test_mae = generate_actual_vs_model_df(model, 
                        X_test, y_test, scaler = None, 
                        trend = None, seasonal = None, index = test_idx, train_on_residuals = False, use_scaler = False)
            
                        
            test_r2 = round(test_r2, 3)
            test_rmse = round(test_rmse, 3)
            test_mae = round(test_mae, 3)
            if kwargs['show_results_plots']:    
                plt.style.use('fivethirtyeight')
                figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')

                ax = plt.subplot()
            

                ax.plot(train_df.Actual, label = 'Training', linewidth = 1, color = 'green')
                ax.plot(test_df.Actual, label = 'Test', linewidth = 1, color = 'black')
                plt.legend(prop={'size': 14})
                plt.title('Training and Testing Data', fontsize = 14)
                plt.show()

                train_title = f"Training: Actual vs. Modeled \n R2: {train_r2}  RMSE: {train_rmse} MAE: {train_mae}"
                args = {'linewidth': 1, 'include_title': False, 'title': train_title, 
                    'include_hline': False,
                    'include_vline': False,}
                plot_data(train_df, args)
                plt.title(train_title, fontsize = 16)
                plt.ylabel(point_name)
                plt.legend(prop={'size': 14})
                plt.show()

                test_title = f"Test: Actual vs. Modeled \n R2: {test_r2}  RMSE: {test_rmse} MAE: {test_mae}"
                args = {'linewidth': 1, 'include_title': False, 'title': test_title, 
                    'include_hline': False,
                    'include_vline': False,}
                plot_data(test_df, args)
                plt.title(test_title, fontsize = 16)  
                plt.ylabel(point_name)
                plt.legend(prop={'size': 14})
                plt.show()
            else:
                print(f"  Training R2: {train_r2}")
                print(f"   Testing R2: {test_r2} \n")
                print(f"Training RMSE: {train_rmse}")
                print(f" Testing RMSE: {test_rmse} \n")
                print(f" Training MAE: {train_mae}")
                print(f"  Testing MAE: {test_mae} \n")

        return  train_df, test_df       
    #return model,X_train, X_test, y_train, y_test, scaler , train_idx, test_idx, data

    else:
        
        print("Model Selected None")
        point_name = kwargs['point']
        train_on_residuals = kwargs['train_on_residuals']        
        
        if train_on_residuals:
            df = decompose_data(df1[point_name], method = kwargs['method'])
            title = f"{point_name}\n Residual Data"
            args = {'linewidth': 2, 'include_title': True, 'title': title, 
            'include_hline': False,
            'include_vline': False,}
            plot_data(df.Noise, args)
            plt.ylabel(f"Residuals = Data - Seasonal - Trend")
        else:
            df = df1.copy()
            df.fillna(method = 'ffill', inplace = True)
            df.fillna(method = 'bfill', inplace = True)
            title = f"{point_name}"
            args = {'linewidth': 2, 'include_title': True, 'title': title, 
            'include_hline': False,
            'include_vline': False,}
            plot_data(df, args)
            plt.ylabel(f"{point_name}")            
                
        
        return df
        #df = append_variables(df)
            
def find_anomalies(df,args, kwargs):
    #model = kwargs['model_type']
    plt.style.use('fivethirtyeight')
    figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
    #if model is not None:
    df = tag_anomalies(df,kwargs = kwargs, method = args['anomalies_method'])
    idx = df.loc[df.Anomalies == 1].index
    method_type = list(args['anomalies_method'].keys())[0]
    threshold = args['anomalies_method'][method_type]
    if kwargs['show_anomalies_plot']:
        figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(df.index,df.Actual, color = 'blue', linewidth = 1, zorder = 0)
        #Area of star = W x H so if we want to make it 3 times bigger we would do (3W) x (3H) = 9 x WH and so on
        s = [20 * 16 for n in range(len(idx))]
        num_anomalies = len(idx)
        plt.scatter(idx,df.loc[df.Anomalies == 1, 'Actual'], label = "Anomalies", color = 'red', marker = '*', linewidth = 2, zorder = 1, s = s, edgecolors = 'white')
        plt.title(f"{num_anomalies} Anomalies Detected\n Using {method_type} Method with {threshold} threshold")
        plt.suptitle(f"{kwargs['point']}", fontsize = 18)
        plt.ylabel(kwargs['point'])
        plt.legend()
        plt.show()
            #df.Anomalies.plot(figsize = (20,10))
        
        #return df
    #else:
        #df = tag_anomalies(df, method = args['anomalies_method'], kwargs = kwargs)

    return df

def drop_anomalies(df, kwargs):
    plt.style.use('fivethirtyeight')
    point_name = kwargs['point']
    initial__shape = df.shape[0]
    num_anomalies = df.loc[df.Anomalies == 1].shape[0]
    new_shape = initial__shape - num_anomalies
    print(f"Number of Points (pre clean): {initial__shape}")
    print(f"Removing {num_anomalies} anomalies...")
    print(f"Number of Points (post clean): {new_shape}" )
    df = df.loc[~(df.Anomalies == 1)]
    if kwargs['show_removed_anomalies_plot']:
        figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
        
        plt.plot(df.index, df.Actual, color = 'blue', linewidth = 1, label= point_name)
        plt.title(f"{point_name}")
        plt.ylabel(f"{point_name}")
        plt.show()
    
    return df

def tag_anomalies(df, kwargs, method = {'iqr' : 3.0}):
    #check that method is correct
    method_type = list(method.keys())[0]
    threshold = method[method_type]
    if  not (method_type == 'iqr' or method_type == 'sd' or method_type == 'percent'):
        raise ValueError("Method needs to be one of iqr or sd.")
    else:
        model = kwargs['model_type']
        if model is not None:
            #get residuals
            df.eval('Anomalies = 0', inplace=True)
            
            if method_type == 'iqr':
                df.eval('Result = Actual - Modeled', inplace=True)
                twenty_fifth = np.percentile(df.Result, 25)
                seventy_fifth = np.percentile(df.Result, 75)
                iqr = seventy_fifth - twenty_fifth
                upper_bound = seventy_fifth + (threshold * iqr)
                lower_bound = twenty_fifth - (threshold * iqr)
                print(f"25:{twenty_fifth}, 75:{seventy_fifth}\n iqr: {iqr} \n up: {upper_bound} lo: {lower_bound}")
                mask = (df.Result > upper_bound) | (df.Result < lower_bound)
                df.loc[mask, 'Anomalies'] = 1
            elif method_type =='sd':
                df.eval('Result = Actual - Modeled', inplace=True)
                #method=SD
                upper_sd = np.mean(df.Result) + threshold * np.std(df.Result)
                lower_sd = np.mean(df.Result) - threshold * np.std(df.Result)
                mask = (df.Result > upper_sd) | (df.Result < lower_sd)
                df.loc[mask, 'Anomalies'] = 1
            else:
                
                df.eval('Result = abs((Actual - Modeled)/Actual * 100)', inplace = True)
                df.loc[df.Result > threshold, 'Anomalies'] = 1
        else:
            #the model is None and can be residuals or just the point itself
            if kwargs['train_on_residuals']:
                df.eval('Anomalies = 0', inplace = True)

                if method_type =='iqr':
                    twenty_fifth = np.percentile(df.Noise, 25)
                    seventy_fifth = np.percentile(df.Noise, 75)
                    iqr = seventy_fifth - twenty_fifth
                    upper_bound = seventy_fifth + (threshold * iqr)
                    lower_bound = twenty_fifth - (threshold * iqr)
                    mask = (df.Noise > upper_bound) | (df.Noise < lower_bound)
                    df.loc[mask, 'Anomalies'] = 1 
                    df = df[['Data', 'Anomalies']]
                    df.rename(columns = {'Data': 'Actual'}, inplace = True)
                else:
                    #method=SD
                    upper_sd = np.mean(df.Noise) + threshold * np.std(df.Noise)
                    lower_sd = np.mean(df.Noise) - threshold * np.std(df.Noise)
                    mask = (df.Noise > upper_sd) | (df.Noise < lower_sd)
                    df.loc[mask, 'Anomalies'] = 1 
                    df = df[['Data', 'Anomalies']] 
                    df.rename(columns = {'Data': 'Actual'}, inplace = True)              
            else:
                #using the data itself without the residuals
                df.eval('Anomalies = 0', inplace = True)
                point_name = kwargs['point']
                if method_type == 'iqr':
                    twenty_fifth = np.percentile(df[point_name], 25)
                    seventy_fifth = np.percentile(df[point_name], 75)
                    iqr = seventy_fifth - twenty_fifth
                    upper_bound = seventy_fifth + (threshold * iqr)
                    lower_bound = twenty_fifth - (threshold * iqr)
                    mask = (df[point_name] > upper_bound) | (df[point_name] < lower_bound)
                    df.loc[mask, 'Anomalies'] = 1   
                    df.rename(columns = {kwargs['point']: 'Actual'}, inplace = True)
                else:
                    #if using 'sd'
                    upper_sd = np.mean(df[point_name]) + threshold * np.std(df[point_name])
                    lower_sd = np.mean(df[point_name]) - threshold * np.std(df[point_name])
                    mask = (df[point_name] > upper_sd) | (df[point_name] < lower_sd)
                    df.loc[mask, 'Anomalies'] = 1    
                    df.rename(columns = {kwargs['point']: 'Actual'}, inplace = True)                                  
            
    return df
    