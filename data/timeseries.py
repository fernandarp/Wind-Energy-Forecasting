import pandas as pd
import numpy as np
import os

class TimeSeries:
    def __init__(self):  
        self.directory = os.path.dirname(os.path.realpath(__file__))
        self.file = 'Database.xlsx'
        self.energy = pd.DataFrame()
        self.temperature = pd.DataFrame()
        self.wind_speed = pd.DataFrame()
        
    def get_energy_data(self, process_data = True):
        if os.path.isdir(self.directory):
            self.energy = pd.read_excel(os.path.join(self.directory, self.file), sheet_name = 'Wind Parks')
            if process_data:
                self.energy = self._process_data(self.energy)
                
        return self.energy
    
    def get_meteorological_data(self, process_data = True):
        if os.path.isdir(self.directory):
            self.temperature = pd.read_excel(os.path.join(self.directory, self.file), sheet_name = 'Meteorological') 
            self.wind_speed['Date'] = self.temperature['Date']
            self.wind_speed['Average Wind Speed'] = self.temperature['Average Wind Speed']
            self.temperature.drop(columns = ['Average Wind Speed'], inplace = True)
            
            if process_data:
                self.temperature = self._process_data(self.temperature)
                self.wind_speed = self._process_data(self.wind_speed)
                
        return self.temperature, self.wind_speed
    
    def _process_data(self, dataframe):
        for column in dataframe.select_dtypes(include = np.number):
            dataframe[column].fillna(dataframe[column].rolling(12, min_periods = 1).mean(), inplace = True)
        return dataframe