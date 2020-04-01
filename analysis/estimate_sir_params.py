"""Credit to: Lewuathe (Github). Source code: https://github.com/Lewuathe/COVID19-SIR"""
import os
import re
import sys
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython.display import display
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

main_dir = os.path.abspath(os.pardir)
sys.path.insert(0, main_dir)

fig_export_path = os.path.join(main_dir, "reports", "estimate_sir_params")
if not os.path.isdir(fig_export_path):
    os.mkdir(fig_export_path)
fig_export_path += os.sep

result_df_vnames = ["country", "beta", "gamma", "r_0"]
country_region_vname = 'Country/Region'
s_0_vname = "s_0"
population_df_vnames = [country_region_vname, s_0_vname]
# beta, gamma = 0.0001, 0.001


class Learner(object):
    def __init__(self, country, loss_funct, df_dct, predict_range, r_0):
        self.country = country
        self.loss = loss_funct
        self.df_dct = df_dct
        self.predict_range = predict_range
        self.r_0 = r_0

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

    def train(self, plot=False):
        # global beta
        # global gamma
        df_dct_tmp, s_0 = self.load_df(self.country)

        recovered = df_dct_tmp[[df_name for df_name in df_dct_tmp.keys() if "recovered" in df_name][0]]
        deaths = df_dct_tmp[[df_name for df_name in df_dct_tmp.keys() if "deaths" in df_name][0]]
        confirmed = df_dct_tmp[[df_name for df_name in df_dct_tmp.keys() if "confirmed" in df_name][0]]
        confirmed = confirmed - recovered - deaths
        i_0 = confirmed.iloc[0]

        # print(self.country, s_0, i_0, self.r_0, confirmed.index)
        # print(len(confirmed) == len(deaths) == len(recovered))
        optimal = minimize(loss, [0.0001, 0.001], args=(confirmed, recovered, s_0, i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        # print(optimal)
        beta, gamma = optimal.x

        if plot:
            # Plotting prep
            new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(
                beta, gamma, confirmed, recovered, deaths, self.country, s_0, i_0, self.r_0)
            df = pd.DataFrame(
                {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death,
                 'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]},
                index=new_index)

            # Actual plotting
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(self.country)
            df.plot(ax=ax)

            # Exporting plot
            export_str = re.sub(r'[^A-Za-z0-9 ]+', '', self.country)
            fig.savefig(fig_export_path+f"{export_str}.png")
            plt.close(fig)

        r_0 = (beta / gamma)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0={r_0:.8f}", end="\n")
        return pd.Series([self.country, beta, gamma, r_0], index=result_df_vnames)

    def predict(self, beta, gamma, confirmed, recovered, deaths, country, s_0, i_0, r_0):
        new_index = self.extend_index(confirmed.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

        extended_actual = np.concatenate((confirmed.values, [None] * (size - len(confirmed.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((deaths.values, [None] * (size - len(deaths.values))))
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(
            SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1))


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]

    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2


def estimate_sir_params(countries, df_dct, predict_range=365, r_0=3.87, plot=False):

    results_df = pd.DataFrame(columns=result_df_vnames)

    for country in countries:
        learner = Learner(country, loss, df_dct, predict_range, r_0)
        results_df = results_df.append(learner.train(plot=plot), ignore_index=True, sort=False)
    return results_df
