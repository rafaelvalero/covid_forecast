import sys
import os
from tqdm import tqdm
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np

from datetime import timedelta
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import Select, Slider
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.models import DatetimeTickFormatter
from bokeh.layouts import column
import math
from bokeh.io import curdoc
sys.path.insert(0,'../covid_forcast')
sys.path.insert(0,'../../../covid_forcast')
# where to save things
# this is for bokey server, to know the location
LOCATION_OF_REPO ='/Users/rafaelvalerofernandez/Documents/repositories/covid_forecast/'
sys.path.insert(0,LOCATION_OF_REPO)
from covid_forecast.utils.data_io import get_data, download_csv_from_link

# where to save things
LOCATION_OF_REPO = ''
OUTPUT = '../outputs/arima'
os.makedirs(OUTPUT,exist_ok=True)
# In case you need to refresh the data, you need a folder /data
download_csv_from_link()

"""List of countries to explore"""
country_list = ['China', 'Italy', 'Germany', 'India', 'Spain', 'United_Kingdom', 'United_States_of_America',
                     'Lithuania', 'Cyprus']

variable_list = ['cases', 'deaths']
data = get_data()
data['dateRep'] = pd.to_datetime(data['dateRep'], infer_datetime_format=True)


def make_dataset(country='Spain', variable='cases', lenght_for_forecast=3):
    data_ = data[data['countriesAndTerritories'] == country].copy()
    data_ = data_.sort_values(by='dateRep')
    # Triming initial zeros
    remove_initia_zeros = np.trim_zeros(data_[variable]).__len__()
    # y = data_[variable][0:remove_initia_zeros]
    y = data_[variable][-remove_initia_zeros:]
    data_labels = data_['dateRep'][-remove_initia_zeros:]
    # taking the last 3. # Change it to andy other amount
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
    return ColumnDataSource(data=dict(data_labels=data_labels,
                forecast=forecasts, conf_int=conf_int,y=y))


def make_plot(source):
    p = figure(plot_width=400, plot_height=400, title = '')
    lenght_for_forecast = source.data['forecast'].__len__()
    p.line(source.data['data_labels'][-lenght_for_forecast:],source.data['forecast'], color='green')
    p.line(source.data['data_labels'][-lenght_for_forecast:],[i[0] for i in source.data['conf_int']], color='red')
    p.line(source.data['data_labels'][-lenght_for_forecast:],[i[1] for i in source.data['conf_int']], color='red')
    p.line(source.data['data_labels'],source.data['y'])
    p.xaxis.formatter=DatetimeTickFormatter(
            hours=["%d %B %Y"],
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],)
    p.xaxis.major_label_orientation = math.pi/2
    p.yaxis.axis_label = '# Cases'
    return p

source = make_dataset(lenght_for_forecast=10, country = 'Germany')
p = make_plot(source)
show(p)



country_ = Select(title="Option:", value="Spain", options=country_list)
variable_ = Select(title="Option:", value="cases", options=variable_list)
lenght_for_forecast_ = Slider(start=1, end=10, value=3, step=1, title="# forecast periods")

# Update function takes three default parameters
def update(attrname, old, new):
    variable = variable_.value
    lenght_for_forecast = lenght_for_forecast_.value
    country = country_.value
    # Updating everythin for user
    data_ = data[data['countriesAndTerritories'] == country].copy()
    data_ = data_.sort_values(by='dateRep')
    # Triming initial zeros
    remove_initia_zeros = np.trim_zeros(data_[variable]).__len__()
    # y = data_[variable][0:remove_initia_zeros]
    y = data_[variable][-remove_initia_zeros:]
    data_labels = data_['dateRep'][-remove_initia_zeros:]
    # taking the last 3. # Change it to andy other amount
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
    source.data =dict(data_labels=data_labels,
                forecast=forecasts, conf_int=conf_int,y=y)


for w in [country_, variable_,lenght_for_forecast_]:
    w.on_change('value', update)

# Set up layouts and add to document
inputs = column(country_, variable_, lenght_for_forecast_)

# To run in the server
curdoc().add_root(column(inputs, p, width=800))
curdoc().title = "ARIMA models"