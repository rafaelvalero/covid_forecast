import shutil
import requests
import pandas as pd


def search_for_xlsx(url='https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide'):
    """
    This search for the xlsx file to download
    :param url: where is the file to dowload
    :return: such as 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-03-20.xlsx'

    """
    response = requests.get(url)
    for i in response.text.split(' '):
        if (i not in ['', ' ']) and (i.__contains__('csv')) and (i.__contains__('https://www.ecdc.europa.eu/sites')):
            x = i
            print(i)
    # TODO: x could generate exception of no assignment
    url_xlsx = x.split('"')[1]
    return url_xlsx


def download_the_data(url='https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide',
    dowload_folder_name='../data/data.xlsx'):
    """
    DEPRECATED AS [European Centre for Disease Prevention and Control](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)
* Script with more cases in `/analysis/playing_with_arima.py`. chage their website.
    Download the data from url and place it in file and folder
    """
    url_xlsx=search_for_xlsx(url=url)
    response = requests.get(url_xlsx, stream=True)
    with open(dowload_folder_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return None


def get_data(dowload_folder_name = '../data/data.csv'):
    data = pd.read_csv(dowload_folder_name, encoding="ISO-8859-1")
    return data


def download_csv_from_link(url='https://opendata.ecdc.europa.eu/covid19/casedistribution/csv',
    dowload_folder_name='../data/data.csv'):
    """
    Download the data from url and place it in file and folder
    """
    response = requests.get(url, stream=True)
    with open(dowload_folder_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return None


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)