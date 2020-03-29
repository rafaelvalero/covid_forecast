"""Credit to: Lewuathe (Github). Source code: https://github.com/Lewuathe/COVID19-SIR"""
#!/usr/bin/python
import os
import sys
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

main_dir = os.path.abspath(os.pardir)
sys.path.insert(0, main_dir)
from analysis import download_data as dd

fig_export_path = os.path.join(main_dir, "reports", "estimate_sir_params"+os.sep)


class Learner(object):
    def __init__(self, country, loss_funct, start_date, predict_range, s_0, i_0, r_0):
        self.country = country
        self.loss = loss_funct
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0

    def load_df(self, filename, country):
        df = pd.read_csv(dd.jh_git_url+filename)
        country_df = df[df['Country/Region'] == country].drop(columns=["Province/State"], errors="ignore")
        return country_df.iloc[0].loc[self.start_date:]

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def train(self):
        recovered = self.load_df(dd.global_recovered_cases_fname, self.country)
        death = self.load_df(dd.global_deaths_cases_fname, self.country)
        confirmed = (self.load_df(dd.global_confirmed_cases_fname, self.country) - recovered - death)

        optimal = minimize(loss, [0.001, 0.001], args=(confirmed, recovered, self.s_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        # print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(
            beta, gamma, confirmed, recovered, death, self.country, self.s_0, self.i_0, self.r_0)
        df = pd.DataFrame(
            {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death,
             'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        fig.savefig(fig_export_path+f"{self.country}.png")
        plt.close(fig)

    def predict(self, beta, gamma, confirmed, recovered, death, country, s_0, i_0, r_0):
        new_index = self.extend_index(confirmed.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

        extended_actual = np.concatenate((confirmed.values, [None] * (size - len(confirmed.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
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


def estimate_sir_params(countries, startdate="1/22/20", predict_range=150, s_0=100000, i_0=2, r_0=10):
    for country in countries:
        learner = Learner(country, loss, startdate, predict_range, s_0, i_0, r_0)
        learner.train()

# def main():
#
#     countries, startdate, predict_range , s_0, i_0, r_0
#
#     for country in countries:
#         learner = Learner(country, loss, startdate, predict_range, s_0, i_0, r_0)
#         #try:
#         learner.train()