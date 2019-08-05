import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.pyplot import figure
figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')

def print_report(df, show_plot = True, separate_plots = False):
    """
    function: prints out a quick overview of each column and how many points are NaN
              and the percentage of NaN values for that column
    params: 
            df- DataFrame, a pandas DataFrame with DateTime index
            show_plot- Boolean, if True it will plot all the columns in a single plot
            seperate_plots- Boolean, if True it will seperate each column and plot 
                           using the subplots option from plot(). This helps in
                           cases where columns have different ranges
    returns:
            If show_plot is True it will return the plot or plots if separate_plots
            is also True, otherwise it only prints the report.
    """
    space = len(max(df.columns, key=len))
    print("\n")
    tot = len("|Percent NaN | Rows NaN/Total Rows")
    print(f"Column{'': <{space-6}}|Percent NaN | Rows NaN/Total Rows")
    print('-'*(space + tot))
    num_rows = df.shape[0]
    for col in df.columns:
            rows_nan = len(df.loc[df[col].isna()])
            
            per = round(rows_nan/num_rows * 100, 2)
            if len(col) == space: 
                print(f"{col}|  {per} %{'':{11-7}}| {rows_nan}/{num_rows}")
            else:
                less_space = space - len(col)
                print(f"{col}{'':{less_space}}|  {per} %{'':{11-7}}| {rows_nan}/{num_rows}")
    if show_plot:
        if not separate_plots:
            df.plot(figsize = (20,10), linewidth = 1)
        else:
            df.plot(subplots = True, figsize = (15,8), linewidth = 1)

def check_kwargs(kwargs):
    if not isinstance(kwargs['point'], str):
        raise TypeError("'point' must be of  type 'str'")
        
    if not (kwargs['training_percent'] > 0.0 and kwargs['training_percent'] < 1.0):
        raise ValueError("'training_percent' must be value between 0.0 and 1.0 exclusive")
        
    if not (kwargs['clean_type'] == 'value' or kwargs['clean_type'] == 'iqr'):
        raise ValueError("'clean_type' must be either 'value' or 'iqr'")
        
    if not (isinstance(kwargs['threshold'], int) or isinstance(kwargs['threshold'], float)):
        raise TypeError("'threshold' must be of type 'int' or 'float'")
        
    if not (kwargs['model_type'] == 'LSTM' or kwargs['model_type'] == 'Random Forest'):
        raise ValueError("'model_type' must be either 'LSTM' or 'Random Forest'")
        
    if not isinstance(kwargs['train_on_residuals'], bool):
        raise TypeError("'train_on_residuals' must be of type 'bool'")
        
    return "Good values!"

#Created by Emma Goldberg
def split_data(data, split = 0.7):
    """
    function: splits the data into a training and testing set
    params: 
          data- A Pandas Series
          split- Float, the percentage user wants to split the training into
                 must be a value 0 < split < 1.0
    returns:
          this functions returns the training and split series into two separate
          Pandas Series
    """
    #check that our data is still a Series
    if isinstance(data, pd.Series) is False:
        raise TypeError("Your data needs to be in a Series format.")
    else:
        #check that split is a float value
        if isinstance(split, float) is False:
            raise TypeError("Your split value needs to be a float.")
        else:
        #check that split is 0.0 < x < 1.0
            if not 0.0 < split < 1.0:
                raise ValueError("Split value needs to be between 0.0 and 1.0")
            else:
                #split data in time format, not randomly shuffled
                length_training = split * len(data)
                length_training = int(round(length_training, 0))
                training = data[0:length_training]
                testing = data[length_training:len(data)]
    return training, testing

#function created by Emma Goldberg
def clean_data(data, threshold, type_clean = 'value', plot=True):
    """
    function: cleans the data using either a one-value threshold method or an iqr method
    params: 
            type_clean: if 'value' is used as the default, uses the threshold method. if 
            another is specified, then the function will use the IQR method.
            plot: shows a plot of the data and the threshold for outliers if used.
    returns:
            the cleaned data as a Series, and a plot of the data with the outlier threshold if specified.
    """
    if type_clean == 'value':
        if isinstance(data, pd.Series) is False:
            raise TypeError("Your data needs to be in a Series format.")
        else:
            #the data is in a Series, check that it is of type numeric
            v = data.values
            is_numeric = np.issubdtype(v.dtype, np.number)
            if is_numeric is False:
                raise TypeError("Your Series data needs to be numeric.")
            else:
            #the data is in the correct format, check that the threshold is an int
                if isinstance(threshold, int) is False:
                    raise TypeError("Your passed threshold needs to be an int.")
                else:
                    #actually clean the data
                    cleaned_data = data[data.values > threshold]
        if plot is True:
            figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(data.index, data.values, label = 'data')
            plt.axhline(y=threshold, color = 'red', linewidth=1)
            plt.xticks(rotation=60)
            plt.xlabel("Time")
            plt.ylabel("Data")
        return cleaned_data
    else:
        #check that the data is a series
        if isinstance(data, pd.Series) is False:
            raise TypeError("Your data needs to be in series format.")
        else:
        #check that their threshold is a float or int
            if isinstance(threshold, int) is False and isinstance(threshold, float) is False:
                raise TypeError("Threshold needs to be either a float or int.")
            else:
                #proceed
                q1 = np.percentile(data,25)
                q3 = np.percentile(data,75)
                iqr = q3 - q1
                lower_bound = q1 -(threshold * iqr)
                upper_bound = q3 +(threshold * iqr) 
                print("lower_bound: %f" %lower_bound,"and","upper_bound: %f" %upper_bound)
                #clean the data
                cleaned_data = data[data.values > lower_bound]
                cleaned_data = data[data.values < upper_bound]
        if plot is True:
            figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(data.index, data.values)
            plt.axhline(y=upper_bound, color = 'red', linewidth=1)
            plt.axhline(y=lower_bound, color = 'red', linewidth=1)
            plt.xticks(rotation=60)
            plt.xlabel("Time")
            plt.ylabel("Data")
        return cleaned_data           

def create_standard_multivariable_df(df, point_location = 0, shift = 1, rename_OAT = True, dropna = True):
    #this function creates a standard 50 variable DataFrame
    """
    variables generated:
        - CDD: Cooling Degree Days (OAT-65), 0 where CDD < 0 
        - HDD: Heating Degree Days (65-OAT), 0 where HDD < 0
        - CDD2: (Cooling Degree Days)^2
        - HDD2: (Heating Degree Days)^2
        - MONTH (1-12): 1 on month of data point, 0 on all other months
        - TOD (0-23): 1 on "TIME OF DAY" of data point, 0 on all other times
        - DOW (0-6): 1 on "DAY OF WEEK" of data point, 0 on all other days
        - WEEKEND: 1 if data point falls on weekend, 0 for everything else
        - Shift_N: user defined shift, where N is the amount of lookback
        - Rolling24_mean: generates the rolling 24hr mean 
        - Rolling24_max: takes the maximun value of the rolling 24hr
        - Rolling 24_min: takes the minimum value of the rolling 24hr
    """
    
    if rename_OAT:
        df.rename(columns={'aiTIT4045':'OAT'}, inplace=True)
    #start_col = len(df.columns)
    df["CDD"] = df.OAT - 65.0
    df.loc[df.CDD < 0 , "CDD"] = 0.0
    df["HDD"] = 65.0 - df.OAT
    df.loc[df.HDD < 0, "HDD" ] = 0.0
    df["CDD2"] = df.CDD ** 2
    df["HDD2"] = df.HDD ** 2
    df.OAT = df.OAT.round(0)
    df.CDD = df.CDD.round(0)
    df.HDD = df.HDD.round(0)
    df.CDD2 = df.CDD2.round(0)
    df.HDD2 = df.HDD2.round(0)
    
    month = [str("MONTH_" + str(x+1)) for x in range(12)]
    df["MONTH"] = df.index.month
    df.MONTH = df.MONTH.astype('category')
    month_df = pd.get_dummies(data = df, columns = ["MONTH"])
    month_df = month_df.T.reindex(month).T.fillna(0)
    month_df = month_df.drop(month_df.columns[0], axis = 1)

    tod = [str("TOD_" + str(x)) for x in range(24)]
    df["TOD"] = df.index.hour
    df.TOD = df.TOD.astype('category')
    tod_df = pd.get_dummies(data = df, columns = ["TOD"])
    tod_df = tod_df.T.reindex(tod).T.fillna(0)
    tod_df = tod_df.drop(tod_df.columns[0], axis = 1)

    dow = [str('DOW_' + str(x)) for x in range(7)]
    df["DOW"] = df.index.weekday
    df.DOW = df.DOW.astype('category')
    dow_df = pd.get_dummies(data = df, columns = ["DOW"])
    dow_df = dow_df.T.reindex(dow).T.fillna(0)
    dow_df = dow_df.drop(dow_df.columns[0], axis = 1)

    df["WEEKEND"] = 0
    df.loc[(dow_df.DOW_5 == 1) | (dow_df.DOW_6 == 1), 'WEEKEND'] = 1

    for i in range(shift):
        shift_col = "SHIFT_" + str(i+1)
        df[shift_col] = df.iloc[ : , point_location].shift(i+1)

    # df["Rolling24_mean"] = df.iloc[ : , point_location].rolling("24h").mean()
    # df["Rolling24_max"] = df.iloc[ : , point_location].rolling("24h").max()
    # df["Rolling24_min"] = df.iloc[ : , point_location].rolling("24h").min()

    df = pd.concat([df, month_df, tod_df, dow_df], axis = 1)
    df.drop(['MONTH', 'TOD', 'DOW'], axis = 1, inplace = True)
    if dropna:
        df.dropna(inplace = True)

    del month_df
    del tod_df
    del dow_df
    #print(f'Generated: {len(df.columns) - start_col} columns')
    return df


def add_variable_to_df(df,fnc,col_name, kwargs):
    
    df[col_name] = fnc(df, **kwargs)
    
    return df
    
def clean_train_data(df, eval_expression = None):
    """
    Used to extract certain values from the DatFrame
    """
    
    if eval_expression:
        evaluated = []
        for exp in eval_expression:
            print(f"Evaluating: {exp}")
            evaluated.append(pd.eval(exp))
        if len(eval_expression) > 1 :
            return tuple(evaluated)
        else:
            return evaluated[0]
    else:
        print("No expression passed")
        return df

# col_to_fill: the column that will be filled from other column's vlaues
# cols_for_fill: the columns that will be used to fill col_to_fill, 
# order of cols will determine how to fill the nans and stale values
def fill_nan_and_stale_values(df,col_to_fill = None, cols_for_fill = None , ffill = False):

    if col_to_fill and isinstance(cols_for_fill, list):
        for col in cols_for_fill:
            print(f"1. Using: {col}")
            df[col_to_fill].fillna(df[col], inplace = True)
        
        df.loc[(df[col_to_fill].pct_change() == 0.0), col_to_fill] = np.nan
        
        for col in cols_for_fill:
            print(f"2. Using: {col}")
            df[col_to_fill].fillna(df[col], inplace = True)
        df.drop(cols_for_fill, axis = 1, inplace = True)
        if ffill:
            return df.ffill()
        else:
            return df
