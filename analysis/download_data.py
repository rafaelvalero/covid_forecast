import sys
from covid_forecast.utils.data_io import download_csv_from_link
sys.path.insert(0, '../../../covid_forcast')


# to download the data, you need to have a data folder
download_csv_from_link()


# Getting data from John Hopking repo
url = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
download_csv_from_link(url=url, dowload_folder_name='../data/data2.csv')