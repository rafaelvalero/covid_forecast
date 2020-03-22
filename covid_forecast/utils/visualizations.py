""" Here the plots and visualizations
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plt_arima_forecast(y, forecasts, conf_int=False,
                       title='Country name here',
                       y_label='Deaths',
                       x=None,
                       save_here='arima_case.png',
                       show_plot = False):
    """

    :param y: real vualues
    :param forecast: predicted values
    :param lenght_for_training: like 90% lenght of y
    :param save_here: str where to save.
    :return:
    """
    lenght_for_forecast = forecasts.__len__()
    plt.clf()
    if x is None:
        x = np.arange(y.shape[0])
    plt.plot(x, y, 'b*--', label='Real')
    plt.plot(x[lenght_for_forecast:], forecasts, 'go--', label='Forecast')
    plt.xlabel('Date')
    plt.title(title)
    plt.ylabel(y_label)
    if conf_int is not False:
        plt.fill_between(x[lenght_for_forecast:],
                         conf_int[:, 0], conf_int[:, 1],
                         alpha=0.1, color='b')
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_here)
    if show_plot:
        plt.show()
    else:
        plt.clf()
    return None

def plt_arima_forecast_outsample(y, forecasts, conf_int=False,
                       title='Country name here',
                       y_label='Deaths',
                       x=None,
                       save_here='arima_case.png',
                       show_plot = False):
    """
    :param y: real vualues
    :param forecast: predicted values
    :param lenght_for_training: like 90% lenght of y
    :param save_here: str where to save.
    :return:
    """
    lenght_for_forecast = forecasts.__len__()
    plt.clf()
    if x is None:
        x = np.arange(y.shape[0])
    plt.plot(x[:y.__len__()], y, 'b*--', label='Real')
    plt.plot(x[-lenght_for_forecast:], forecasts, 'go--', label='Forecast')
    plt.xlabel('Date')
    plt.title(title)
    plt.ylabel(y_label)
    if conf_int is not False:
        plt.fill_between(x[-lenght_for_forecast:],
                         conf_int[:, 0], conf_int[:, 1],
                         alpha=0.1, color='b')
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_here, dpi=1000)
    if show_plot:
        plt.show()
    else:
        plt.clf()
    return None

def render_pic_in_notebook(location_file = '../outputs/arima/forecast_next_3days_Spain_Deaths.png',
                           set_size_inches=[19,9]):
    """making notebook more visual"""
    img = mpimg.imread(location_file)
    # from now on you can use img as an image, but make sure you know what you are doing!
    imgplot=plt.imshow(img)
    plt.gcf().set_size_inches(set_size_inches[0], set_size_inches[1])
    plt.axis('off')
    plt.show()