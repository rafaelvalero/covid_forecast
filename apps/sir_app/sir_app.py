"""
To run this app
you shoud go to the folder and run
'''bokeh serve --show sir_v3.py'''
"""

from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure

from bokeh.models import CategoricalColorMapper, HoverTool, ColumnDataSource, Panel
from bokeh.models.widgets import CheckboxGroup, Slider, RangeSlider, Tabs

from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_16

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
import pandas as pd
import numpy as np
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource, Div
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, output_file, show
from bokeh.io import curdoc, show, output_notebook
import os
import scipy.integrate

# to upload some text for the users
homepage = Div(text=open(os.path.join(os.getcwd(), 'homepage.html')).read(), width=800)
explanation_sir_model = Div(text=open(os.path.join(os.getcwd(), 'explanation_sir_model.html')).read(), width=800)

def SIR_model(y, t, betta, gamma):
    """
    Tradiional SIR model.
    :param y:
    :param t:
    :param betta:
    :param gamma:
    :return:
    """
    S,I,R = y
    dS_dt = -betta*S*I
    dI_dt = betta*S*I - gamma*I
    dR_dt = gamma*I
    return ([dS_dt, dI_dt, dR_dt])

# def make_dataset(amplitude, offset, phase, freq):
def make_dataset(S0=0.9, I0=0.1, R0 = 0, betta=0.35, gamma=0.1):
    """
    Create the data to be ploted
    :param S0:
    :param I0:
    :param R0:
    :param betta:
    :param gamma:
    :return:
    """
    # Get the current slider values
    # time vector
    t = np.linspace(0, 100, 10000)
    # Result
    solution = scipy.integrate.odeint(SIR_model, [S0, I0, R0], t, args=(betta, gamma))
    solution = np.array(solution)
    return ColumnDataSource(data=dict(t=t, S=solution[:,0], I=solution[:,1], R=solution[:,2]))


def make_plot(source):
    """Creation of simple graphs in bokeh"""
    # Set up plot
    p = figure(plot_width=400, plot_height=400)
    #p.vline_stack(['S', 'R', 'I'], x='t', source=source)
    p.vline_stack(['S'], x='t', source=source, color='blue',  width=4)
    p.vline_stack(['I'], x='t', source=source, color='red', width=4)
    p.vline_stack(['R'], x='t', source=source, color='green', width=4)
    return p
# Create first grapsh
source = make_dataset()
p = make_plot(source)

# Set up widgets
text = TextInput(title="title", value='My SIR model')
gamma_ = Slider(title="Gamma", value=0.1, start=0.0, end=2.0, step=0.01)
betta_ = Slider(title="Beta", value=0.35, start=0.0, end=2.0, step=0.01)

# Update function takes three default parameters
def update(attrname, old, new):
    """Notice this function does not retunr anything but modify the
    data for the graph"""
    # Get the current slider values
    t = np.linspace(0, 100, 10000)
    S0 = 0.9
    I0 = 0.1
    R0 = 0
    betta = betta_.value
    gamma = gamma_.value
    # Result
    solution = scipy.integrate.odeint(SIR_model, [S0, I0, R0], t, args=(betta, gamma))
    solution = np.array(solution)
    # Generate the new curve
    source.data = dict(t=t, S=solution[:,0], I=solution[:,1], R=solution[:,2])

# Updating everythin for user
for w in [gamma_, betta_]:
    w.on_change('value', update)


def update_title(attrname, old, new):
    """Upadate title of the raphs"""
    p.title.text = text.value
text.on_change('value', update_title)



# Set up layouts and add to document
inputs = column(text, gamma_, betta_)
layout = ([[homepage], [inputs, p], [explanation_sir_model]])

# To run in the server
curdoc().add_root(column(homepage, inputs, p , explanation_sir_model, width=800))
curdoc().title = "SIR models"
