# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure


class Orchestrator:
    NOBS = 5
    DUMMY_LEVELS = pd.DataFrame({
        't': 1 + np.arange(NOBS),
        'x': np.clip(10 + np.random.randn(NOBS), 8, 13),
        'y': np.clip(1 + .1 * np.random.randn(NOBS), 0, .2)
    })
    DUMMY_SDS = pd.DataFrame({
        'x': np.linspace(.2, .3, NOBS),
        'y': np.linspace(.01, .02, NOBS),
    })

    def __init__(self):
        self.source = self.init_source()
        self.plot = self.init_plot()

    def init_plot(self) -> figure:
        plot = figure(height=400, width=400, title="my sine wave",
                      tools="crosshair,pan,reset,save,wheel_zoom",
                      x_range=[0, 4 * np.pi], y_range=[-2.5, 2.5])

        plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)
        return plot

    def init_source(self) -> ColumnDataSource:
        source = ColumnDataSource(data=self.DUMMY_LEVELS+0)
        return source


orchestrator = Orchestrator()

# Set up plot



def make_title_widget():
    text = TextInput(title="title", value='my sine wave')



offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4*np.pi, N)
    y = a*np.sin(k*x + w) + b

    source.data = dict(x=x, y=y)

for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)

inputs = column(text, offset, amplitude, phase, freq)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "double-layer of sliders"


__author__ = 'Petr Panov'
__copyright__ = 'Copyleft 2022, Milky Way'
__credits__ = ['Petr Panov']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Petr Panov'
__email__ = 'pvpanov93@gmail.com'
__status__ = "Draft"
