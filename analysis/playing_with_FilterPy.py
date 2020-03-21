"""
There are a number of works related to Kalman Filter (hereafter KF)
Here to explore a bit different python KF implementations, such as FilterPy: https://filterpy.readthedocs.io/en/latest/#
Interesting paper about KF: https://arxiv.org/pdf/1204.0375.pdf implementation in Python

:TODO
Continue working on the estimation of the parameters of the KF.
"""
import sys, os
from covid_forecast.utils.data_io import get_data, download_the_data
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import numpy as np
# borrow this library from the book from rlabbe
from covid_forecast.utils.book_plots import plot_measurements
from numpy.random import randn
from tqdm import tqdm


sys.path.insert(0,'../../../covid_forcast')
# where to save things
OUTPUT = '../outputs/playing_with_FilterPy'
os.makedirs(OUTPUT,exist_ok=True)
# In case you need to refresh the data / you need a folder /data
# download_the_data()
"""To save some time just run the part you want"""
run_example = False
run_real_cases = True
"""
Case from:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/08-Designing-Kalman-Filters.ipynb
"""
if run_example:
    dt = 1.
    R = 3.
    kf = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
    kf.P *= 10
    kf.R *= R
    kf.Q = Q_discrete_white_noise(2, dt, 0.1)
    kf.F = np.array([[1., 0], [0., 0.]])
    kf.B = np.array([[dt], [ 1.]])
    kf.H = np.array([[1., 0]])
    print(kf.P)
    zs = [i + randn()*R for i in range(1, 100)]
    xs = []
    cmd_velocity = 1.
    for z in zs:
        kf.predict(u=cmd_velocity)
        kf.update(z)
        xs.append(kf.x[0])
    plt.clf()
    plt.plot(xs, label='Kalman Filter')
    plot_measurements(zs)
    plt.xlabel('time')
    plt.legend(loc=4)
    plt.ylabel('distance');
    plt.savefig(OUTPUT+'/example_filterpy.png')
    plt.clf()
"""Real case"""
if run_real_cases:
    data = get_data()
    # Only one country for the moment
    #country='United_Kingdom'
    #variable = 'Cases'
    #for country in data['Countries and territories'].unique():
    #    for variable in ['Cases','Deaths']:
    for country in tqdm(['China', 'Spain', 'United_Kingdom', 'United_States', 'Cyprus']):
        for variable in ['Cases', 'Deaths']:
            data_ = data[data['Countries and territories']==country].copy()
            data_ = data_.sort_values(by='DateRep')
            zs = data_[variable]
            zs = np.trim_zeros(zs)
            dt = 1.
            R = 3.
            kf = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
            kf.P *= 10
            kf.R *= R
            kf.Q = Q_discrete_white_noise(2, dt, 0.1)
            kf.F = np.array([[1., 0], [0., 0.]])
            kf.B = np.array([[dt], [ 1.]])
            kf.H = np.array([[1., 0]])
            xs = []
            cmd_velocity = 1.
            for z in zs:
                kf.predict(u=cmd_velocity)
                kf.update(z)
                xs.append(kf.x[0])
            plt.clf()
            plt.plot(xs, label='Kalman Filter')
            plot_measurements(zs)
            plt.xlabel('time')
            plt.legend(loc=4)
            plt.title(country)
            plt.ylabel(variable)
            plt.savefig(OUTPUT+'/example_filterpy_{}_{}.png'.format(country ,variable))
            plt.clf()
