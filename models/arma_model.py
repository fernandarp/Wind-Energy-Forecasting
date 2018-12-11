import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import arma_order_select_ic

class ARMAModel():
    def __init__(self, data, test_size):
        self.test_size = test_size
        self.train = data[:len(data)-test_size]
        self.test = data[len(data)-test_size:]
        
    def run_ARMA(self):
        self.arma_order = arma_order_select_ic(self.train, 5, 5)
        
        model = ARMA(self.train, order=self.arma_order['bic_min_order'])        
        model_fit = model.fit(transparams=False)
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        
        predictions = model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1, dynamic=False)
        
        return np.array(predictions)
    
    def run_ARMAX(self, exogenous_data):
        self.armax_order = arma_order_select_ic(self.train, 5, 5)
        self.exogenous_train = exogenous_data[:len(exogenous_data)-self.test_size]
        self.exogenous_test = exogenous_data[len(exogenous_data)-self.test_size:]
        
        model = ARMA(self.train, order=self.armax_order['bic_min_order'], exog = self.exogenous_train)
        model_fit = model.fit()
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        
        predictions = model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1, dynamic=False, exog=self.exogenous_test)
        
        return np.array(predictions)