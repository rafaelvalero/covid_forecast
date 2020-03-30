"""
Let's check this glorious repo from R in Python
Some explanation here https://youtu.be/10pvXLKw5dQ

Blogs:
1) Basic explanation ARIMA https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/
3) Comprehensive example is you would like to see what is ARIMA:
 https://datafai.com/auto-arima-using-pyramid-arima-python-package/
Library:
1) https://pypi.org/project/pmdarima/
2) http://alkaline-ml.com/pmdarima/0.9.0/setup.html
"""
import sys
import os
from covid_forecast.utils.data_io import get_data, download_csv_from_link
from covid_forecast.utils.visualizations import plt_arima_forecast,plt_arima_forecast_outsample
from tqdm import tqdm
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd

sys.path.insert(0,'../../../covid_forcast')
# where to save things
OUTPUT = '../outputs/arima/france_departments'
os.makedirs(OUTPUT,exist_ok=True)
# In case you need to refresh the data, you need a folder /data
#download_csv_from_link()
"""To save some time just run the part you want"""
DATA = '../data/france/donnees-hospitalieres-covid19-2020-03-28-19h17.csv'
METADATA = '../data/france/department_number_name_region.xlsx'
lenght_for_forecast = 3
data = pd.read_csv(DATA, sep=';')

#data['dep'] = data['dep'].astype(int)
print(data.head())
metadata = pd.read_excel(METADATA, sep=';')
print(metadata.head())

metadata.head()

metadata['INSEE code']= metadata['INSEE code'].astype(str)
metadata['INSEE code'][0:9]= '0'+metadata['INSEE code'][0:9]

data = pd.merge(metadata, data, left_on='INSEE code', right_on='dep')

axes_info = {'hosp':'People in Hospital',
             'rea':'Critical Care',
             'rad':'Return at Home',
             'dc':'Deaths in Hospital'}

DATE_COLUMNS_NAME = 'jour'
COUNTRY_COLUMNS = 'Department'

list_variables = ['hosp','rea','rad','dc']
country_list= data[COUNTRY_COLUMNS].unique()

data.head().T

data[DATE_COLUMNS_NAME] = pd.to_datetime(data[DATE_COLUMNS_NAME], infer_datetime_format=True)
data = data.groupby(['Department','jour']).sum()
data = data.reset_index()
report_country = pd.DataFrame()
report = pd.DataFrame()
for country in tqdm(country_list):
    print('Working on: {}'.format(country))
    first_variable = pd.DataFrame()
    for variable in list_variables:
        try:
            data_ = data[data[COUNTRY_COLUMNS] == country].copy()
            data_ = data_.sort_values(by=DATE_COLUMNS_NAME)
            # Triming initial zeros
            remove_initia_zeros = np.trim_zeros(data_[variable]).__len__()
            # y = data_[variable][0:remove_initia_zeros]
            y = data_[variable][-remove_initia_zeros:]
            data_labels = data_[DATE_COLUMNS_NAME][-remove_initia_zeros:]
            # taking the last 3. # Change it to any other amount
            # Fit your model
            model = pm.auto_arima(y, seasonal=False, suppress_warnings=True)
            # make your forecasts
            # predict N steps into the future
            forecasts, conf_int = model.predict(lenght_for_forecast, return_conf_int=True)
            # Adding labels for each new day
            data_labels = data_labels.to_list()
            for i in range(1, lenght_for_forecast + 1):
                data_labels.append(data_labels[-1] + timedelta(1))
            forecasts, conf_int = model.predict(lenght_for_forecast, return_conf_int=True)
            # Visualize the forecasts (blue=train, green=forecasts)
            plt_arima_forecast_outsample(y, forecasts, conf_int=conf_int,
                                         title=country,
                                         y_label=axes_info[variable],
                                         x=data_labels,
                                         save_here=OUTPUT + '/forecast_next_3days_{}_{}.png'.format(country, variable))
            # To save the data
            df_for_data = pd.DataFrame()
            df_for_data = pd.DataFrame(y.to_list() + forecasts.tolist(),
                                       columns=[variable])
            df_for_data[COUNTRY_COLUMNS] = country
            df_for_data[DATE_COLUMNS_NAME] = data_labels
            if first_variable.empty:
                first_variable = df_for_data
            else:
                first_variable = first_variable.merge(df_for_data, on=(DATE_COLUMNS_NAME, COUNTRY_COLUMNS))
        except Exception as e:
            print(e)
    if report.empty:
        report = first_variable
    else:
        report = pd.concat([report, first_variable])
if report_country.empty:
    report_country = report
else:
    report_country = pd.concat([report_country, report])
# Creation of report
report_country.to_csv(OUTPUT + "/forecast_next_free_days.csv")
