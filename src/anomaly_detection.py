import configparser
from enum import Enum
import re
import os
import sys
import ast
import pandas as pd
from matplotlib import pyplot as plt 
from src.model_generator import *
from .data_helper import *
cwd = os.getcwd()
if cwd[-3:] == 'src':
    config_path = cwd + '/config/lstmconfig.ini'
else:
    config_path = cwd + '/src/config/lstmconfig.ini'
config = configparser.ConfigParser()
something = config.read(config_path)
#iqr multiplier will be {'type': [lower_multiplier, upper_multiplier] }
multiplier = {'STEAM': [2.0, 6.0], 'CHILLEDWATER' : [2.0, 2.0], 'ELECTRICITY': [2.0, 2.0], 'OTHER' : [3.0, 2.0]}

class Tag(Enum):
    STEAM = 1
    CHILLEDWATER = 2
    ELECTRICITY = 3
    OTHER = 4

class AnomalyDetection:
    def __init__(self):
        self.point = None
        self.point_type = None
        self.pc = None
        self._add_eco_tools()
        self.point_data = pd.DataFrame()
        self.model_data = pd.DataFrame()
        self.results_data = pd.DataFrame()
        self.cleaned_data = pd.DataFrame()
        self.kwargs = None
        self._load_kwargs()
        self.interval = config['PI']['interval']
        self.calculation = config['PI']['calculation']
        self.start = config['PI']['start']
        self.end = config['PI']['end']
        self.training_percent = float(config['model']['train_percent'])
        self.epochs = int(config['model']['epochs'])

    def _add_eco_tools(self):
        eco_tools_path = config['SETUP']['eco_tools_path']
        sys.path.append(eco_tools_path)
        from ecotools.pi_client import pi_client
        self.pc = pi_client(root = 'readonly')
    
    def _load_kwargs(self):
        cwd = os.getcwd()
        kawrgs_path = cwd + '/src/kwargs.txt'
        with open(kawrgs_path, 'r') as infile:
            self.kwargs = ast.literal_eval(infile.read())


    def find_point(self, point_name, show_points_found = True):
        points = self.pc.search_by_point(point_name)
        self.point = points[0]
        self._get_point_type()
        print(f'Point Used: {self.point}')
        print(f"Point Type: {self.point_type.name}\n")
        if show_points_found:
            print(f"Points Found:")
            for i in range(len(points)):
                print(f"{i+1}- {points[i]}")

    def _get_point_type(self):
        if re.search('steam', self.point, re.IGNORECASE):
            self.point_type = Tag.STEAM
        elif re.search('chilledwater', self.point, re.IGNORECASE):
            self.point_type = Tag.CHILLEDWATER
        elif re.search('electricity', self.point, re.IGNORECASE):
            self.point_type = Tag.ELECTRICITY
        else:
            self.point_type = Tag.OTHER
        
    
    def download_point_data(self):
        self.point_data = self.pc.get_stream_by_point(self.point, start = self.start, end = self.end, interval = self.interval, calculation = self.calculation)
        print_report(self.point_data)
        self.point_data.dropna(inplace = True)

    def clean_data(self):
        self.kwargs.update({'point': self.point})
        self.model_data = split_and_clean(self.point_data, self.kwargs)

    def run_neural_network(self, show_neural_network_running = False, epochs = None):
        if not(epochs == None) and isinstance(epochs, int):
            self.kwargs.update({'epochs':epochs})

        if show_neural_network_running:
            _, self.results_data = create_model(self.model_data, self.kwargs)
        else:
            print('Running Model...')
            import contextlib
            #from io import 
            @contextlib.contextmanager
            def capture():
                #import sys
                from io import StringIO
                oldout,olderr = sys.stdout, sys.stderr
                try:
                    out=[StringIO(), StringIO()]
                    sys.stdout,sys.stderr = out
                    yield out
                finally:
                    sys.stdout,sys.stderr = oldout, olderr
                    out[0] = out[0].getvalue()
                    out[1] = out[1].getvalue()

            with capture() as out:
                _, self.results_data = create_model(self.model_data, self.kwargs)
        self.results_data.eval('Result = Modeled - Actual', inplace = True)
        self._get_anomalies()
    
    def remove_anomalies(self):
        self.cleaned_data = self.results_data.loc[self.results_data['Anomalies'] == 0, 'Actual']
        self.cleaned_data.rename(columns = {'Actual': self.point}, inplace = True)
        figure(num=None, figsize=(20,5), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(self.cleaned_data.index, self.cleaned_data, linewidth= 1, label = self.point)
        plt.ylabel(self.point)
        plt.title(f"Cleaned Data")
        plt.legend()
    def save(self, which = None):
        from datetime import datetime
        date = datetime.now().strftime('%m%d%y %H%M')
        if which is None:
            self.results_data.to_csv(f'{self.point}_Anomalies_{date}.csv')
        else:
            self.results_data.loc[:, which].to_csv(f'{self.point}_Anomalies_{date}.csv')

    def _get_anomalies(self):
        days = int(self.kwargs['days'])
        odd = True if days % 2 == 1 else False
        shift_multiplier = int(days / 2) if odd else int(days / 2) - 1
        q75 = self.results_data.Result.rolling(24*days).quantile(0.75).shift(-(24*shift_multiplier))
        q25 = self.results_data.Result.rolling(24*days).quantile(0.25).shift(-(24*shift_multiplier))
        iqr = q75 - q25
        lower_multiplier = multiplier[self.point_type.name][0]
        upper_multiplier = multiplier[self.point_type.name][1]
        
        lower_threshold = q25 - lower_multiplier * iqr
        upper_threshold = q75 + upper_multiplier * iqr
        upper_threshold.fillna(method = 'bfill', inplace = True)
        upper_threshold.fillna(method = 'ffill', inplace = True)
        lower_threshold.fillna(method = 'bfill', inplace = True)
        lower_threshold.fillna(method = 'ffill', inplace = True)
        
        upper_mask = (self.results_data.Result > upper_threshold)
        lower_mask = (self.results_data.Result < lower_threshold)
        mask = upper_mask | lower_mask
        self.results_data.eval('Anomalies = 0', inplace = True)
        self.results_data.loc[mask, 'Anomalies'] = 1
        num_anomalies = len(self.results_data.loc[self.results_data['Anomalies'] == 1].index)
        s1 = [20 * 9 for n in range(num_anomalies)]
        
        figure(num=None, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
        
        plt.plot(self.results_data.index, self.results_data.Actual, linewidth= 1, label = self.point, zorder = 1)
        plt.scatter(self.results_data.loc[self.results_data['Anomalies'] == 1].index,
                self.results_data.loc[self.results_data['Anomalies'] == 1, 'Actual'], marker = "X", 
                color ="red",linewidth = 2, zorder = 2, s = s1, edgecolors = 'white', label = 'Anomalies')

        plt.ylabel(self.point)
      
        total = num_anomalies
        percent = float(total / self.results_data.shape[0]) * 100.0
        plt.title(f"Total Anomalies: {total} ({round(percent, 2)}%)")
        plt.legend()


