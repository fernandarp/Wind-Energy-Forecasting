# Time Series Forecasting with AR

# Links: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import arma_order_select_ic

class ARModel():
    def __init__(self, data, test_size):
        self.test_size = test_size
        self.train = data[:len(data)-test_size]
        self.test = data[len(data)-test_size:]
        
    def run_AR(self):
        model = AR(self.train)
        model_fit = model.fit()
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        
        predictions = model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1, dynamic=False)
        
        return np.array(predictions)
    
    def run_ARX(self, exogenous_data):        
        self.arx_order = arma_order_select_ic(self.train, 10, 0)
        self.exogenous_train = exogenous_data[:len(exogenous_data)-self.test_size]
        self.exogenous_test = exogenous_data[len(exogenous_data)-self.test_size:]
        
        model = ARMA(self.train, order=self.arx_order['bic_min_order'], exog = self.exogenous_train)
        model_fit = model.fit()
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        
        predictions = model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1, dynamic=False, exog=self.exogenous_test)
        
        return np.array(predictions)