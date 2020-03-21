"""
Let's check this glorious repo from R in Python

Blogs:
1) Basic explanation ARIMA https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/
3) Comprehensive example is you would like to see what is ARIMA:
 https://datafai.com/auto-arima-using-pyramid-arima-python-package/

Library:
1) https://pypi.org/project/pmdarima/
2) http://alkaline-ml.com/pmdarima/0.9.0/setup.html

:TODO
Incorporate predictions for tomorrow. 1) Graphs and 2) tables


"""
import sys
import os
from covid_forecast.utils.data_io import get_data, download_the_data
from covid_forecast.utils.visualizations import plt_arima_forecast
from tqdm import tqdm
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'../../../covid_forcast')
# where to save things
OUTPUT = '../outputs/arima'
os.makedirs(OUTPUT,exist_ok=True)
# In case you need to refresh the data, you need a folder /data
# download_the_data()
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
    for country in tqdm(['China','Italy', 'Germany','India', 'Spain', 'United_Kingdom', 'United_States', 'Cyprus']):
    #for country in tqdm(['United_Kingdom']):
        print('Working on: {}'.format(country))
        for variable in ['Cases', 'Deaths']:
 #       for variable in ['Cases']:
            try:
                data_ = data[data['Countries and territories']==country].copy()
                data_ = data_.sort_values(by='DateRep')
                # Triming initial zeros
                remove_initia_zeros = np.trim_zeros(data_[variable]).__len__()
                #y = data_[variable][0:remove_initia_zeros]
                y = data_[variable][-remove_initia_zeros:]
                data_labels = data_['DateRep'][-remove_initia_zeros:]
                # taking 90% of the data
                lenght_for_training = round(y.__len__()*0.9)
                # taking the last 3
                #lenght_for_training = 4
                train, test = train_test_split(y, train_size=lenght_for_training)
                # Fit your model
                model = pm.auto_arima(train, seasonal=False, suppress_warnings=True)
                # make your forecasts
                #forecasts = model.predict(test.shape[0])  # predict N steps into the future
                forecasts, conf_int =model.predict(test.shape[0], return_conf_int=True)
                # Visualize the forecasts (blue=train, green=forecasts)
                plt_arima_forecast(y, forecasts,conf_int=conf_int,
                                   lenght_for_training=lenght_for_training,
                                   title=country,
                                   y_label=variable,
                                   x=data_labels,
                                   save_here=OUTPUT + '/arima_{}_{}.png'.format(country, variable))

            except Exception as e: print(e)

country = 'Spain'
case = 'Cases'

