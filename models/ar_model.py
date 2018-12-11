# Time Series Forecasting with AR

# Links: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR

class ARModel():
    def __init__(self, data, test_size):
        self.train = data[:len(data)-test_size]
        self.test = data[len(data)-test_size:]
        
    def run_AR(self):
        model = AR(self.train)
        model_fit = model.fit()
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        
        predictions = model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1, dynamic=False)
        
        return np.array(predictions)