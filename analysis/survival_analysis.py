import sys
import os
from tqdm import tqdm
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
sys.path.insert(0, os.path.abspath('../../../covid_forecast'))
from covid_forecast.utils.data_io import get_data, download_the_data
from covid_forecast.utils.visualizations import plt_arima_forecast,plt_arima_forecast_outsample, render_pic_in_notebook


# where to save things
OUTPUT = '../outputs/survival_analysis'
#With 1085 people 42 deaths
#DATA_LOCATTION = '../data/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv'
# Below file contains more cases
#DATA_LOCATTION = '../data/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv'
# Data from SK, https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv
DATA_LOCATTION = '../data/coronavirusdataset/PatientInfo.csv'
os.makedirs(OUTPUT, exist_ok=True)

data = pd.read_csv(DATA_LOCATTION)
data = data[[i for i in data.columns if not i.__contains__('Unnamed')]]
data.head().T


"""Check numbers death, recovered and sick
For file COVID19_line_list_data.csv
"""
try:
    print('Size sample: {}'.format(data.shape[0]))
    print('Number casualties {}'.format((data['death'] == '1').sum()))
    print('Number recovered {}'.format((data['recovered'] == '1').sum()))
    print('Sick People (non recovered, non death) {}'.format(((data['death'] != '1') & (data['recovered'] != '1')).sum()))
except Exception as e: print(e)
"""Check numbers death, recovered and sick
For file COVID19_line_list_data.csv
"""
try:
    print('Size sample: {}'.format(data.shape[0]))
    print('Number casualties {}'.format((data['death'] == '1').sum()))
    print('Number recovered {}'.format((data['recovered'] == '1').sum()))
    print('Sick People (non recovered, non death) {}'.format(((data['death'] != '1') & (data['recovered'] != '1')).sum()))
except Exception as e: print(e)
"""Check numbers death, recovered and sick
# Data from SK, https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv

"""
try:
    print('Size sample: {}'.format(data.shape[0]))
    print('Number casualties {}'.format((data['state'] == 'deceased').sum()))
    print('Number recovered {}'.format((data['state'] == 'released').sum()))
except Exception as e: print(e)

"""Features"""
data['confirmed_date'] = pd.to_datetime(data['confirmed_date'])
data['released_date'] = pd.to_datetime(data['released_date'])
data['symptom_onset_date'] = pd.to_datetime(data['symptom_onset_date'])
data['deceased_date'] = pd.to_datetime(data['deceased_date'])
#data['duraction_confirmed_death'] = data['released_date']-data['confirmed_date']
data['duraction_death_confirmed'] = data['deceased_date']-data['confirmed_date']
data['duraction_death_symptons'] = data['deceased_date']-data['symptom_onset_date']

