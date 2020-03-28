import os
import sys

main_dir = os.path.abspath(os.pardir)
sys.path.insert(0, main_dir)
from covid_forecast.utils.data_io import download_csv_from_link

# folder path and download url path
download_foldname = os.path.join(main_dir, "data", "raw")
jh_git_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'

# Filenames
global_confirmed_cases = "time_series_covid19_confirmed_global.csv"
global_deaths_cases = "time_series_covid19_deaths_global.csv"
global_recovered_cases = "time_series_covid19_recovered_global.csv"

all_files = [global_confirmed_cases, global_deaths_cases, global_recovered_cases]

# Make directory if not already there
if not os.path.isdir(download_foldname):
    os.mkdir(download_foldname)
download_foldname += os.sep

# Getting data from John Hopkins repo
for file in all_files:
    download_csv_from_link(url=jh_git_url+file, dowload_folder_name=download_foldname+file)
