
import dash
import visdcc
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_cytoscape as cyto

#import dash_core_components as dcc
#import dash_html_components as html
#from dash.dependencies import Input, Output
#import plotly.express as px
#import pandas as pd
import pathlib
from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

# owner: shivp Kaggle. Source: https://data.mendeley.com/datasets
# dataset was modified. Original data: https://www.kaggle.com/shivkp/customer-behaviour
dfg = pd.read_csv(DATA_PATH.joinpath("topic_words_all.csv"))

dfg['word'] = dfg['word'].astype(str)
dfg['topic'] = dfg['topic'].astype(str)
dfg['type'] = dfg['type'].astype(str)
dfg = dfg[(dfg['type'] == "harddrugs")]

terms = ['All', '2018', '2019', '2020', '2021']
terms1 = ['All','Hard Drugs', 'Rape', 'Abuse', 'Burglary']

node_list = list(set(dfg['word'].unique().tolist() + \
                     dfg['topic'].unique().tolist())
                )
nodes = [{'id': node_name, 'label': node_name, 'shape': 'dot', 'size': 7}
        for i, node_name in enumerate(node_list)]

edges = []
for row in dfg.to_dict(orient='records'):
    source, target, weight = row['word'], row['topic'], row['value']
    edges.append({
        'id': source + "__" + target,
        'from': source,
        'to': target,
        'width': 2,
        'height': 10
    })

layout = html.Div([
    html.H1('LSA Word Topics', style={"textAlign": "center"}),

    html.H5('Please select a year', style={"textAlign": "left"}),
        html.Div(dcc.Dropdown(
            id='genre-dropdown', value='2018', clearable=False,
            options=[{'label': x, 'value': x} for x in terms],
        ), className='six columns'),

        html.H5('Please select a crime type', style={"textAlign": "left"}),

        html.Div(dcc.Dropdown(
            id='sales-dropdown', value='Rape', clearable=False,
            persistence=True, persistence_type='memory',
            options=[{'label': x, 'value': x} for x in terms1],
        ), className='six columns'),

    visdcc.Network(id = 'net',
                  data = {'nodes': nodes, 'edges': edges},
                  options = dict(height= '600px', width= '100%')),
    dcc.RadioItems(id = 'color',
                  options=[{'label': 'Red', 'value': '#ff0000'},
                          {'label': 'Green', 'value': '#00ff00'},
                          {'label': 'Blue', 'value': '#0000ff'}],
                  value='Blue' )
    
])

@app.callback(
    Output('net', 'options'),
     [Input('color', 'value')]
)
def myfun(x):
    #return 'You have selected "{}"'.format(value)
    return {'nodes':{'color': x}}
