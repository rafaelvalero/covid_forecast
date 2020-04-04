"""

Sandbox

Idea:
    1) create report for all countries in list with the graph normalized and no-normalized.
    2) Create picts
    3) Optimatize
        a) with RN  instead of gamma and beta at the same time. In this case (beta, RN). So beta and RN and not
            gamma
            # RN is reproduction number
            see https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
        b) by betta and gamma
    4) Collect of the imfor creating a report
    5) Select the best option by country

Everyting Self contain in this script


References:
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
import sys
import json
import urllib.request
import os
import shutil

"""At the top the functions after the report"""

def remove_province(input_file, output_file):
    input = open(input_file, "r")
    output = open(output_file, "w")
    output.write(input.readline())
    for line in input:
        if line.lstrip().startswith(","):
            output.write(line)
    input.close()
    output.close()


def download_data(url_dictionary, DATA_FOLDER = "./data" ):
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
    def __init__(self, country, loss,s_0 = 1, i_0=1, r_0=1, predict_range=300, DATA_FOLDER="./data", verbose=True):
        self.country = country
        self.DATA_FOLDER = DATA_FOLDER
        self.verbose = verbose
        self.loss = loss
        #self.start_date = start_Data
        self.start_date = self.get_starting_date()
        print(self.start_date )
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
    def  get_starting_date(self):
        """
        This function help you with the starting date
        :param country:
        :return:
        """
        df = pd.read_csv(self.DATA_FOLDER+'/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        # Remove values no related to date
        country_df = country_df.iloc[:, 4:]
        length_array = country_df.shape[1]
        remove_initial_zeros = np.trim_zeros(country_df.iloc[:, 4:].values[0]).__len__()
        # Select start date as well
        return country_df.iloc[:, length_array - remove_initial_zeros].name

    def load_confirmed(self):
        df = pd.read_csv(self.DATA_FOLDER+'/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:]

    def load_recovered(self):
        df = pd.read_csv(self.DATA_FOLDER+'/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:]

    def load_dead(self):
        df = pd.read_csv(self.DATA_FOLDER+'/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:]

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, country, s_0, i_0, r_0, normalized=True):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            if normalized:
                return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
            else:
                return [-beta * S * I / s_0, beta * S * I / s_0 - gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, \
               extended_death, solve_ivp(SIR, [0, size],
                             [s_0, i_0, r_0],
                             t_eval=np.arange(0, size, 1),method='LSODA')


    def train(self, normalized=True, solver_method='L-BFGS-B',minimiazing_RN=False ):
        """
        This functions
        1) prepare the data
        2) set up initial values accordingly
        3) Run the solver
        4) collect information: like the algorithms and so on
        5) Create graphs

        :param normalized: boolean. Is the model normalized?
        :param solver_method: str. Solver name to minimize the loss function to fit data
        :param minimiazing_RN:  boolean. Do you want maximazing using (beta, RN) or (beta, gamma)
        :return: the solver info, where are the files locations
        """

        self.recovered = self.load_recovered()
        self.death = self.load_dead()
        self.data = (self.load_confirmed() - self.recovered - self.death)

        # Set up initial values and bonds
        if normalized:
            initial_values = [0.2/self.s_0, 0.1/self.s_0]
            bounds = [(0.00000001, 0.2), (0.00000001, 0.1)]
        else:
            initial_values = [0.2, 0.1]
            bounds = [(0.01, 2), (0.001, 2)]
        if minimiazing_RN:
            initial_values = [initial_values[0], 2]
            bounds = [bounds[0], (1, 5)]

        # Compute the minimum
        optimal = minimize(loss,initial_values,
                           args=(self.data, self.recovered, self.s_0, self.i_0,
                                 self.r_0, int(normalized), int(minimiazing_RN)),
                           method=solver_method, bounds=bounds)
        optimal.update({'method': solver_method})
        if self.verbose:
            print(optimal)
        # Checking for what case of minimization are we
        if minimiazing_RN:
            # RF is reproduction number
            beta, RN = optimal.x
            gamma = beta / RN
        else:
            beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death,\
                                prediction = self.predict(beta, gamma, self.data,
                                                          self.recovered, self.death,
                                                          self.country,
                                                          self.s_0, self.i_0,
                                                          self.r_0,
                                                          normalized=normalized)
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
        filename = '{}_normalized_{}_RN_{}'.format(self.country,normalized,minimiazing_RN)
        filename_location = OUTPUT+"/{}.png".format(filename)
        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 7))
        # :TODO put a more informative name
        #ax.set_title(self.country)
        ax.set_title(filename)
        df.plot(ax=ax)
        fig.savefig(filename_location)
        optimal.update({'filename_location':filename_location,
                        'filename':filename})
        if plot_grpahs:
            plt.gcf()
            plt.show()
        else:
            plt.close('all')
        if self.verbose:
            print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")

        return optimal


def loss(point, data, recovered, s_0, i_0, r_0, normalized, minimiazing_RN):
    """

    :param point:
    :param data:
    :param recovered:
    :param s_0:
    :param i_0:
    :param r_0:
    :param normalize: boolean this is to normalize or not the problem
    :return:
    """
    size = len(data)
    if bool(minimiazing_RN):
        # RF is reproduction number
        beta, RN = point
        gamma = beta/RN
    else:
        beta, gamma = point

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        if bool(normalized):
            return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
        else:
            return [-beta * S * I / s_0, beta * S * I / s_0 - gamma * I, gamma * I]

    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1),
                         vectorized=True,method='LSODA')
                         #vectorized=True)

    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered) ** 2))
    # :TODO to check properly this parameter, what is doing here?
    alpha = 0.9
    return alpha * l1 + (1 - alpha) * l2

def selection_of_best_models_and_move_pics(report_results, maximum_treshold_for_RN = 100):
    """
    There are different cases less select the best one some how
    :param report_results: dataframe: the report from all cases.
    :param maximum_treshold_for_RN: I see there are some final graphs with very high RN, a threshold for that
    :return:
    """
    for country in report_results['country']:
        # Country cases
        slice_report_by_country = report_results[report_results['country'] == country]
        # Check in if the country has models with any RN reasonable
        # :TODO perhaps easy rank it?
        slice_report_by_country = slice_report_by_country[slice_report_by_country['RN']\
                                                          < maximum_treshold_for_RN ]
        # Select the cases with less error and give me the location
        best_graphs_location =slice_report_by_country.sort_values('fun')['filename_location'].head(1).values[0]
        # Create copy with different name
        # get the path
        location_file = best_graphs_location.split('/')
        # Change the name
        new_location_fie = '/'.join(location_file[:-1]) + '/BEST_{}.png'.format(country)
        shutil.copy(best_graphs_location, new_location_fie)

if __name__ == '__main__':
    """This will allow to use this as a library later.
    The doctest is for start introducing variables if wanted"""
    import doctest
    doctest.testmod(verbose=True)

    """Here some parameters to play around for the creation of the report"""
    #countries = ['United Kingdom', 'Spain', 'US', 'Italy', 'Cyprus', 'France', 'India']
    countries = ['Spain', 'United Kingdom']
    download = False
    plot_grpahs = False
    # Plaug here some countries population
    # :TODO George have this data in functions
    countries_populations = {'United Kingdom': 66488991,
                             'Spain': 46723749,
                             'US': 327167434,
                             'Italy': 60431283,
                             'Cyprus': 1189265,
                             'France': 66900000,
                             'China': 1386000000,
                             'India': 1339000000}
    OUTPUT = '../outputs/sir_models'
    DATA_FOLDER = '../data/jh_data'
    [os.makedirs(i, exist_ok=True) for i in [DATA_FOLDER, OUTPUT]]
    """------------------------------------------------"""
    # In case there is no data, folder for the data or need and update
    if download:
        data_d=load_json("../data/data_url.json")
        download_data(data_d, DATA_FOLDER=DATA_FOLDER)

    remove_province(DATA_FOLDER+'/time_series_19-covid-Confirmed.csv', DATA_FOLDER+'/time_series_19-covid-Confirmed-country.csv')
    remove_province(DATA_FOLDER+'/time_series_19-covid-Recovered.csv', DATA_FOLDER+'/time_series_19-covid-Recovered-country.csv')
    remove_province(DATA_FOLDER+'/time_series_19-covid-Deaths.csv', DATA_FOLDER+'/time_series_19-covid-Deaths-country.csv')

    results = []
    for country in countries:
        info = {}
        learner = Learner(country, loss, s_0=countries_populations[country], i_0=10, r_0=5, DATA_FOLDER=DATA_FOLDER)
        # try:
        for normalized in [True, False]:
            optimal = learner.train(normalized=normalized, solver_method='L-BFGS-B')
            for minimiazing_RN in [True, False]:
                optimal = learner.train(normalized=normalized, solver_method='L-BFGS-B', minimiazing_RN=minimiazing_RN)
                # Collecting metadata
                info = {}
                info['country'] = country
                info.update(optimal)
                info['SIR normalized'] = normalized
                info['minimazing RE'] = minimiazing_RN
                if minimiazing_RN:
                    beta, RN = optimal.x
                    info['beta'] = beta
                    info['gamma'] = beta/RN
                    info['RN'] = RN
                else:
                    beta, gamma = optimal.x
                    info['beta'] = beta
                    info['gamma'] = gamma
                    info['RN'] = beta / gamma
                results.append(info)


    # Create report
    report_results = pd.DataFrame(results)
    report_results.to_csv(OUTPUT+'/models_metadata_v6.csv')
    """point out the best pics for country in the output folder"""
    selection_of_best_models_and_move_pics(report_results)


result_df_vnames = ["country", "beta", "gamma", "RN"]
maximum_treshold_for_RN = 100



def selection_of_best_models_and_move_pics(report_results, maximum_treshold_for_RN = 100,
                                           result_df_vnames = ["country", "beta", "gamma", "RN"]):
    """
    There are different cases less select the best one some how
    :param report_results: dataframe: the report from all cases.
    :param maximum_treshold_for_RN: I see there are some final graphs with very high RN, a threshold for that
    :return:
    """
    list_results = []
    for country in report_results['country']:
        # Country cases
        slice_report_by_country = report_results[report_results['country'] == country]
        # Check in if the country has models with any RN reasonable
        # :TODO perhaps easy rank it?
        slice_report_by_country = slice_report_by_country[slice_report_by_country['RN'] \
                                                          < maximum_treshold_for_RN]
        # Select the cases with less error and give me the location
        best_graphs_location = slice_report_by_country.sort_values('fun')[result_df_vnames].head(1).values[0]
        info = {}
        for index_ in range(result_df_vnames.__len__()):
            print(index_)
            info[result_df_vnames[index_]] = best_graphs_location[index_]
        list_results.append(info)
    return pd.DataFrame(list_results)




        return