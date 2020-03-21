"""
Let's check this glorious repo from R in Python

Blogs:
1) https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/


Library
1) https://pypi.org/project/pmdarima/
2) http://alkaline-ml.com/pmdarima/0.9.0/setup.html
"""
import pandas as pd
import sys
import os
from covid_forecast.utils.data_io import get_data
from tqdm import tqdm
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'../../../covid_forcast')
# where to save things
OUTPUT = '../outputs/arima'
os.makedirs(OUTPUT,exist_ok=True)
"""To save some time just run the part you want"""
run_example = False
run_real_cases = True

if run_example:
    """Example"""
    # Author: Taylor Smith <taylor.smith@alkaline-ml.com
    # Load/split your data
    y = pm.datasets.load_wineind()
    train, test = train_test_split(y, train_size=150)

    # Fit your model
    model = pm.auto_arima(train, seasonal=True, m=12)

    # make your forecasts
    forecasts = model.predict(test.shape[0])  # predict N steps into the future

    # Visualize the forecasts (blue=train, green=forecasts)
    plt.clf()
    x = np.arange(y.shape[0])
    plt.plot(x[:150], train, c='blue')
    plt.plot(x[150:], forecasts, c='green')
    plt.savefig(OUTPUT+'/arima_example.png')
    plt.show()
    plt.clf()

if run_real_cases:
    """Real data ARIMA"""
    data = get_data()
    for country in tqdm(['China', 'Spain', 'United_Kingdom', 'United_States', 'Cyprus']):
        print('Working on: {}'.format(country))
        for variable in ['Cases', 'Deaths']:
            try:
                data_ = data[data['Countries and territories']==country].copy()
                data_ = data_.sort_values(by='DateRep')
                y = data_[variable]
                # taking 90% of the data
                lenght_to_predict = round(y.__len__()*0.9)
                train, test = train_test_split(y, train_size=lenght_to_predict)
                # Fit your model
                model = pm.auto_arima(train, seasonal=False)
                # make your forecasts
                forecasts = model.predict(test.shape[0])  # predict N steps into the future
                # Visualize the forecasts (blue=train, green=forecasts)
                plt.clf()
                x = np.arange(y.shape[0])
                plt.plot(x, y, c='blue', label='Real')
                plt.plot(x[lenght_to_predict:], forecasts, c='green', label='Forecast')
                plt.legend()
                plt.xlabel('time')
                plt.title(country)
                plt.ylabel(variable)
                plt.savefig(OUTPUT+'/arima_{}_{}.png'.format(country, variable))
                plt.clf()
            except Exception as e: print(e)

