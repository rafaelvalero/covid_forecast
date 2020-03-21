""" Here the plots and visualizations
"""
import matplotlib.pyplot as plt
import numpy as np

def plt_arima_forecast(y, forecasts, conf_int=False,
                       lenght_for_training=5,
                       title='Country name here',
                       y_label='Deaths',
                       x=None,
                       save_here='arima_case.png'):
    """

    :param y: real vualues
    :param forecast: predicted values
    :param lenght_for_training: like 90% lenght of y
    :param save_here: str where to save.
    :return:
    """
    plt.clf()
    if x is None:
        x = np.arange(y.shape[0])
    plt.plot(x, y, 'b*--', label='Real')
    plt.plot(x[lenght_for_training:], forecasts, 'go--', label='Forecast')
    plt.xlabel('time')
    plt.title(title)
    plt.ylabel(y_label)
    if conf_int is not False:
        plt.fill_between(x[lenght_for_training:],
                         conf_int[:, 0], conf_int[:, 1],
                         alpha=0.1, color='b')
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_here)
    plt.clf()
    return None