import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic

class ARIMAModel():
    def __init__(self, data, test_size):
        self.test_size = test_size
        self.train = data[:len(data)-test_size]
        self.test = data[len(data)-test_size:]
        
    def run_ARIMA(self, order):
        model = ARIMA(self.train, order=order)        
        model_fit = model.fit(transparams=False)
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        
        predictions = model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1, dynamic=False)
        
        return np.array(predictions)