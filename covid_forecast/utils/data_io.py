import sys, os
import csv
import requests
from urllib.parse import urlparse

import shutil
import requests



def search_for_xlsx(url = 'https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide'):
    """
    This search for the xlsx file to download
    :param url: where is the file to dowload
    :return: such as 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-03-20.xlsx'

    """
    response = requests.get(url)
    for i in response.text.split(' '):
        if (i not in ['', ' ']) and i.__contains__('.xlsx'):
            x = i
            print(i)
    url_xlsx = x.split('"')[1]
    return url_xlsx

def download_the_data(url = 'https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide',
    dowload_folder_name = '../../data/data.xlsx'):
    """
    Download the data from url and place it in file and folder
    """
    url_xlsx=search_for_xlsx(url=url)
    response = requests.get(url_xlsx, stream=True)
    with open('../../data/data.xlsx', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return None


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)