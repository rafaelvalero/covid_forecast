import sys
from covid_forecast.utils.data_io import download_the_data
sys.path.insert(0,'../../../covid_forcast')

# to download the data, you need to have a data folderd
download_the_data()