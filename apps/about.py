import plotly.graph_objects as go
import pandas as pd

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#from app import app

# needed if running single page dash app instead
external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("About the project", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='Blablabla. '
                                     )
                    , className="mb-4")
            ]),

        dbc.Row([
            dbc.Col(html.H5(children='Blablabla.')
                    , className="mb-5")
        ]),
    ])
])
