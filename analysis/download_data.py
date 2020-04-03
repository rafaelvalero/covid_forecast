import os
import sys

main_dir = os.path.abspath(os.pardir)
sys.path.insert(0, main_dir)
from covid_forecast.utils.data_io import download_csv_from_link

# folder path and download url path
download_foldname = os.path.join(main_dir, "data", "raw", "time_series")
export_foldname = os.path.join(main_dir, "data", "processed")
jh_git_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
population_data_url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"

# Filenames
global_confirmed_cases_fname = "time_series_covid19_confirmed_global.csv"
global_deaths_cases_fname = "time_series_covid19_deaths_global.csv"
global_recovered_cases_fname = "time_series_covid19_recovered_global.csv"

all_files = [global_confirmed_cases_fname, global_deaths_cases_fname, global_recovered_cases_fname]


# Make directory if not already there
def safe_mkdir(abs_dir_path):
    """
    Safely makes new directories if not alredy there

    Args:
        abs_dir_path (str): of absolute path required

    Returns:
        str: of target directory
    """
    if not os.path.isdir(abs_dir_path):
        os.makedirs(abs_dir_path, exist_ok=True)
    abs_dir_path += os.sep
    return abs_dir_path


download_foldname = safe_mkdir(download_foldname)
export_foldname = safe_mkdir(export_foldname)


# Getting data from John Hopkins repo
for file in all_files:
    download_csv_from_link(url=jh_git_url+file, dowload_folder_name=download_foldname+file)

#abs_dir_path=download_foldname

#if not os.path.isdir(abs_dir_path):
#    os.mkdir(abs_dir_path)