"""Credit to: Lewuathe (Github). Source code: https://github.com/Lewuathe/COVID19-SIR"""
import os
# import re
import sys
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython.display import display
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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


main_dir = os.path.abspath(os.pardir)
sys.path.insert(0, main_dir)

fig_export_path = os.path.join(main_dir, "reports", "estimate_sir_params")
if not os.path.isdir(fig_export_path):
    os.mkdir(fig_export_path)
fig_export_path += os.sep

result_df_vnames = ["country", "beta", "gamma", "RN"]
country_region_vname = 'Country/Region'
s_0_vname = "s_0"
population_df_vnames = [country_region_vname, s_0_vname]
beta, gamma = 0.001, 0.001


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

class Learner(object):
    def __init__(self, country, loss, df_dct, s_0=1, i_0=1, r_0=1, predict_range=300, DATA_FOLDER=fig_export_path, verbose=True):

        self.country = country
        self.loss = loss
        self.df_dct = df_dct
        self.predict_range = predict_range
        self.DATA_FOLDER = DATA_FOLDER
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.verbose = verbose

    def load_df(self, country):
        confirmed_df_name = "confirmed_global"
        df_dct_tmp = self.df_dct.copy()
        vars_to_use = df_dct_tmp[list(df_dct_tmp.keys())[0]].drop(columns=country_region_vname)
        for df_name in df_dct_tmp.keys():
            if "confirmed" in df_name:
                df_dct_tmp[df_name] = df_dct_tmp[df_name][df_dct_tmp[df_name][country_region_vname] == country]
                s_0 = int(df_dct_tmp[df_name][s_0_vname].iloc[0])
                df_dct_tmp[df_name] = df_dct_tmp[df_name].iloc[0].drop(index=[country_region_vname, s_0_vname])
                vars_to_use = np.trim_zeros(df_dct_tmp[df_name], trim="fb").index  # trims head and tail of series 0s
                df_dct_tmp[df_name] = df_dct_tmp[df_name][vars_to_use]
                confirmed_df_name = df_name
                break
        for df_name in [df_name_tmp for df_name_tmp in df_dct_tmp.keys() if df_name_tmp != confirmed_df_name]:
            df_dct_tmp[df_name] = df_dct_tmp[df_name][df_dct_tmp[df_name][country_region_vname] == country]
            df_dct_tmp[df_name] = df_dct_tmp[df_name].iloc[0].drop(index=[country_region_vname])
            df_dct_tmp[df_name] = df_dct_tmp[df_name][vars_to_use]
        return df_dct_tmp, s_0

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

    def train(self, normalized=True, solver_method='L-BFGS-B',minimiazing_RN=False,plot_option=False):
        df_dct_tmp, self.s_0 = self.load_df(self.country)

        self.recovered = df_dct_tmp[[df_name for df_name in df_dct_tmp.keys() if "recovered" in df_name][0]]
        self.death = df_dct_tmp[[df_name for df_name in df_dct_tmp.keys() if "deaths" in df_name][0]]
        confirmed = df_dct_tmp[[df_name for df_name in df_dct_tmp.keys() if "confirmed" in df_name][0]]
        self.confirmed = confirmed - self.recovered - self.death
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
        if self.verbose:
            print(self.country, "population:", self.s_0, "start date", confirmed.index[0])  #, "infected:", i_0)
        # Compute the minimum
        optimal = minimize(loss,initial_values,
                           args=(self.confirmed, self.recovered, self.s_0, self.i_0,
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

        new_index, extended_actual, extended_recovered, extended_death, \
        prediction = self.predict(beta, gamma, self.confirmed,
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

        # Plotting prep
        # Changes the dates format
        df.reset_index(inplace=True)
        df['dates'] = pd.to_datetime(df['index'])
        del df['index']
        df.set_index('dates', inplace=True)
        filename = '{}_normalized_{}_RN_{}'.format(self.country, normalized, minimiazing_RN)
        filename_location = self.DATA_FOLDER + "/{}.png".format(filename)
        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 7))
        # :TODO put a more informative name
        # ax.set_title(self.country)
        ax.set_title(filename)
        df.plot(ax=ax)
        #fig.savefig(fig_export_path+f"{self.country}.png")
        fig.savefig(filename_location)
        optimal.update({'filename_location': filename_location,
                        'filename': filename})
        if plot_option:
            plt.gcf()
            plt.show()
        else:
            plt.close('all')
        plt.close(fig)


        RN = beta / gamma
        if self.verbose:
            print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0={RN:.8f}", "\n")
        return pd.Series([self.country, beta, gamma, RN], index=result_df_vnames), optimal



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

def selection_of_best_models_information(report_results, maximum_treshold_for_RN = 100,
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
            info[result_df_vnames[index_]] = best_graphs_location[index_]
    list_results.append(info)
    return pd.DataFrame(list_results)

def estimate_sir_params(countries, df_dct, predict_range=365, i_0=2, r_0=0, plot_option=False):

    results_df = pd.DataFrame(columns=result_df_vnames)

    for country in countries:
        learner = Learner(country, loss, df_dct, predict_range, i_0, r_0)
        results_df, optimal = learner.train(plot_option=plot_option)
    return results_df


def estimate_sir_params_v2(countries, df_dct, predict_range=365, i_0=2,
                           r_0=0, plot_option=True):

    results_df = pd.DataFrame(columns=result_df_vnames)
    results = []
    for country in countries:
        learner = Learner(country, loss, df_dct, predict_range=predict_range,
                      i_0=i_0, r_0=r_0,)
        for normalized in [True, False]:
            for minimiazing_RN in [True, False]:
                results_df, optimal = learner.train(normalized=normalized, solver_method='L-BFGS-B',
                    minimiazing_RN=minimiazing_RN, plot_option=plot_option)
                # Collecting metadata
                info = {}
                info['country'] = country
                info.update(optimal)
                info['SIR normalized'] = normalized
                info['minimazing RE'] = minimiazing_RN
                if minimiazing_RN:
                    beta, RN = optimal.x
                    info['beta'] = beta
                    info['gamma'] = beta / RN
                    info['RN'] = RN
                else:
                    beta, gamma = optimal.x
                    info['beta'] = beta
                    info['gamma'] = gamma
                    info['RN'] = beta / gamma
                results.append(info)
    report_results = pd.DataFrame(results)
    if plot_option:
        selection_of_best_models_and_move_pics(report_results)
    # Create summary best plots
    # :TOdo Add population?
    results_df = selection_of_best_models_information(report_results, maximum_treshold_for_RN = 100,
                            result_df_vnames = ["country", "beta", "gamma", "RN"])
    return results_df, report_results
