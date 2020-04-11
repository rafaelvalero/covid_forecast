import sys
import os
from tqdm import tqdm
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np

from datetime import timedelta
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import Select, Slider, Div, ColumnDataSource, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, grid
import math
from bokeh.io import curdoc
sys.path.insert(0,'../covid_forcast')
sys.path.insert(0,'../../../covid_forcast')
# where to save things
# this is for bokey server, to know the location
LOCATION_OF_REPO ='/Users/rafaelvalerofernandez/Documents/repositories/covid_forecast/'
sys.path.insert(0,LOCATION_OF_REPO)
from covid_forecast.utils.data_io import get_data, download_csv_from_link

from bokeh.io import output_file, show
from bokeh.models import Dropdown
# where to save things
LOCATION_OF_REPO = ''
OUTPUT = '../outputs/arima'
os.makedirs(OUTPUT,exist_ok=True)
# In case you need to refresh the data, you need a folder /data
dowload_folder_name='../../data/data2.csv'
#download_csv_from_link(dowload_folder_name=dowload_folder_name)


"""List of countries to explore"""


variable_list = ['cases', 'deaths']
data = get_data(dowload_folder_name=dowload_folder_name)
#country_list = ['China', 'Italy', 'Germany', 'India', 'Spain', 'United_Kingdom', 'United_States_of_America',
#                     'Lithuania', 'Cyprus']
country_list = data.countriesAndTerritories.unique().tolist()
data['dateRep'] = pd.to_datetime(data['dateRep'], format='%d/%m/%Y')
"""Some documentation"""
homepage = Div(text=open(os.path.join(os.getcwd(), 'homepage.html')).read(), width=800)
explanation_sir_model = Div(text=open(os.path.join(os.getcwd(), 'explanation_model.html')).read(), width=800)

""" Key functions """
tooltips = [('forecast','@forecast'),
            ('Real Values', '@y')]


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
    # Make all series equali long
#    conf_int_upper_bound = np.append(y,conf_int[:,1])
#    conf_int_lower_bound = np.append(y,conf_int[:,0])
    nan_array_ = np.empty(y.__len__()) * np.nan
    conf_int_upper_bound = np.append(nan_array_, conf_int[:, 1])
    conf_int_lower_bound = np.append(nan_array_, conf_int[:, 0])
    forecasts = np.append(nan_array_,forecasts)
    # Adding labels for each new day
    data_labels = data_labels.to_list()
    y = y.to_list()
    for i in range(1, lenght_for_forecast + 1):
        data_labels.append(data_labels[-1] + timedelta(1))
        y.append(np.nan)
    return ColumnDataSource(data=dict(data_labels=data_labels,
                forecast=forecasts,
                conf_int_upper_bound=conf_int_upper_bound,
                conf_int_lower_bound = conf_int_lower_bound,
                y=y))


def make_plot(source,country = 'Spain', variable = 'cases'):
    p = figure(plot_width=800, plot_height=800, title = country, tooltips=tooltips)
    lenght_for_forecast = source.data['forecast'].__len__()
    x = source.data['data_labels'][-lenght_for_forecast:]
    y = source.data['y']
    upper_band =source.data['conf_int_upper_bound']
    lower_band = source.data['conf_int_lower_bound']
    # Bands are drawn as patches. That is, a polygon specified by a series of 2D points
    # Because points are specified in clockwise order, the lower band needs to be reverse (Hence the [::-1])
    xs = np.concatenate([x, x[::-1]])
    ys = np.concatenate([upper_band, lower_band[::-1]])
    p.line(x,source.data['forecast'], color='blue', legend_label = 'Forecast')
    p.circle(x = 'data_labels', y ='forecast', source = source, fill_color="blue", size=8)
    p.line(x,upper_band, color='blue')
    p.line(x,lower_band, color='blue')
    p.line(x, y, legend_label = 'Real Values',color='black')
    p.circle(x = 'data_labels', y = 'y' ,source = source, fill_color="black", size=8)
    p.patch(x=xs, y=ys, fill_alpha=0.3, line_alpha=0, legend_label="Interval of Confidence")
    p.xaxis.formatter=DatetimeTickFormatter(
            days=["%d/%m/%Y"])
    p.xaxis.major_label_orientation = math.pi/2
    p.yaxis.axis_label = '# of new {}'.format(variable)
    p.legend.location = 'top_left'
    return p

# Update function takes three default parameters
def update(attrname, old, new):
    variable = variable_.value
    lenght_for_forecast = lenght_for_forecast_.value
    country = country_.value
    # Updating everythin for user
    #print('updating {} {} {}'.format(variable,lenght_for_forecast,country))
    new_source = make_dataset(country=country, variable=variable, lenght_for_forecast=lenght_for_forecast)
    #source.data.update(new_source)
    #source.data = dict(new_source.data)
    source.data.update(new_source.data)
    #print(source.data['y'])
    layout.children[2] = make_plot(source, country=country, variable=variable)

# Set up layouts and add to document
country_ = Select(title="Option:", value="Spain", options=country_list)
country_.on_change('value', update)
variable_ = Select(title="Option:", value="cases", options=variable_list)
variable_.on_change('value', update)
lenght_for_forecast_ = Slider(start=1, end=10, value=3, step=1, title="# forecast days")
lenght_for_forecast_.on_change('value', update)

source = make_dataset(lenght_for_forecast=3, country = 'Spain', variable = "cases")
inputs = column(country_,variable_, lenght_for_forecast_)
p = [make_plot(source)]
layout= column(homepage,inputs,*p, explanation_sir_model)
curdoc().add_root(layout)

