"""
https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model
https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/


ORIGINAL CODE from:
  Lewuathe (Github).  https://github.com/Lewuathe/COVID19-SIR
"""

# !/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request
import os


"""Here some parameters to play around"""
countries = ['United Kingdom','Spain','US','Italy','Cyprus', 'France', 'India']
download = True
plot_grpahs = True
countries_populations = {'United Kingdom':66488991,
                         'Spain':46723749 ,
                         'US': 327167434,'Italy':60431283 ,
                         'Cyprus':1189265,
                         'France':66900000,
                         'China': 1386000000,
                         'India': 1339000000}
OUTPUT = 'output'
DATA_FOLDER = '../data/jh_data'
[os.makedirs(i,exist_ok=True) for i in [DATA_FOLDER]]
"""------------------------------------------------"""

def remove_province(input_file, output_file):
    input = open(input_file, "r")
    output = open(output_file, "w")
    output.write(input.readline())
    for line in input:
        if line.lstrip().startswith(","):
            output.write(line)
    input.close()
    output.close()


def download_data(url_dictionary, DATA_FOLDER = "./data/" ):
    # Lets download the files
    for url_title in url_dictionary.keys():
        if DATA_FOLDER.endswith('/'):
            urllib.request.urlretrieve(url_dictionary[url_title], DATA_FOLDER + url_title)
        else:
            urllib.request.urlretrieve(url_dictionary[url_title], DATA_FOLDER + '/'+ url_title)


def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)


class Learner(object):
    def __init__(self, country, loss,s_0 = 1, i_0=1, r_0=1, predict_range=300,DATA_FOLDER="./data"):
        self.country = country
        self.loss = loss
        #self.start_date = start_Data
        self.start_date = self.get_starting_date()
        print(self.start_date )
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.DATA_FOLDER = DATA_FOLDER
    def  get_starting_date(self, DATA_FOLDER = DATA_FOLDER):
        """
        This function help you with the starting date
        :param country:
        :return:
        """
        df = pd.read_csv(DATA_FOLDER+'/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        # Remove values no related to date
        country_df = country_df.iloc[:, 4:]
        length_array = country_df.shape[1]
        remove_initial_zeros = np.trim_zeros(country_df.iloc[:, 4:].values[0]).__len__()
        # Select start date as well
        return country_df.iloc[:, length_array - remove_initial_zeros].name

    def load_confirmed(self, country, DATA_FOLDER = DATA_FOLDER):
        df = pd.read_csv(DATA_FOLDER+'/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]

    def load_recovered(self, country, DATA_FOLDER = DATA_FOLDER):
        df = pd.read_csv(DATA_FOLDER+'/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]

    def load_dead(self, country, DATA_FOLDER = DATA_FOLDER):
        df = pd.read_csv(DATA_FOLDER+'/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, country, s_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            #return [-beta * S * I / s_0, beta * S * I /s_0 - gamma * I, gamma * I]
            return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, \
               extended_death, solve_ivp(SIR, [0, size],
                             [s_0, i_0, r_0],
                             t_eval=np.arange(0, size, 1),method='LSODA')
                             #t_eval=np.arange(0, size, 1), method='LSODA')

    def train(self):
        self.recovered = self.load_recovered(self.country)
        self.death = self.load_dead(self.country)
        self.data = (self.load_confirmed(self.country) - self.recovered - self.death)

        optimal = minimize(loss, [0.2/self.s_0, 0.1/self.s_0], args=(self.data, self.recovered, self.s_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 2), (0.00000001, 0.1)])
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death,\
                                prediction = self.predict(beta, gamma, self.data,
                                                          self.recovered, self.death,
                                                          self.country,
                                                          self.s_0, self.i_0,
                                                          self.r_0)
        df = pd.DataFrame(
            {'Infected data': extended_actual, 'Recovered data': extended_recovered,
             'Death data': extended_death,
             'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]},
            index=new_index)
        # Changes the dates format
        df.reset_index(inplace=True)
        df['dates'] = pd.to_datetime(df['index'])
        del df['index']
        df.set_index('dates', inplace=True)
        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(self.country)
        df.plot(ax=ax)
        fig.savefig(f"output/{self.country}.png")
        if plot_grpahs:
            plt.gcf()
            plt.show()
        else:
            plt.close('all')
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")

        return df


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        #return [-beta * S * I / s_0, beta * S * I /s_0 - gamma * I, gamma * I]
        return [-beta * S * I , beta * S * I  - gamma * I, gamma * I]

    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1),
                         vectorized=True,method='LSODA')
                         #vectorized=True)

    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered) ** 2))
    alpha = 0.9
    return alpha * l1 + (1 - alpha) * l2



if download:
    data_d=load_json("../data/data_url.json")
    download_data(data_d, DATA_FOLDER=DATA_FOLDER)

remove_province(DATA_FOLDER+'/time_series_19-covid-Confirmed.csv', DATA_FOLDER+'/time_series_19-covid-Confirmed-country.csv')
remove_province(DATA_FOLDER+'/time_series_19-covid-Recovered.csv', DATA_FOLDER+'/time_series_19-covid-Recovered-country.csv')
remove_province(DATA_FOLDER+'/time_series_19-covid-Deaths.csv', DATA_FOLDER+'/time_series_19-covid-Deaths-country.csv')

for country in countries:
    learner = Learner(country, loss, s_0=countries_populations[country], i_0=10, r_0=5, DATA_FOLDER=DATA_FOLDER)
    # try:
    df = learner.train()

df = pd.read_csv(DATA_FOLDER+'/time_series_19-covid-Deaths-country.csv')