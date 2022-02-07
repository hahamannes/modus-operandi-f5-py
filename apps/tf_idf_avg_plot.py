
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import pathlib
from app import app
import plotly.io as pio

pio.templates.default = "presentation"


# --------------------- Packages

import re
import numpy as np
import pandas as pd
from pprint import pprint

from datetime import datetime

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
# import pyLDAvis.gensim  # Author: don't skip this
# pyLDAvis.gensim.prepare

# I think i need another one:
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvis

# Plots
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# TF.IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('dutch')
stop_words.extend(['tenlastelegging', 'hof', 'althans', 'tenlastegelegd', 'naan', 'verklaring', 'verklaren', 'benadelen', 'naam', 'aangeefster', 'aangever', 'aangev',
 'verbalisant', 'slachtoffer', 'rechtbank', 'uur', 'uren', 'weten', 'bestaan', 'waarheid', 'daarvoor', 'genaamd', 'maken', 'gaan', 'toverweging', 'aanzien', 'bewijs', 'feit', 
 'grond', 'staan', 'vaststellen', 'halen', 'vervolgens', 'nemen', 'aanhouden', 'bevinden', 'officier', 'justitie', 'overtuigen', 'bewijzen', 'maken', 'stellen', 'leggen', 'dienen', 
 'vrijspreken', 'daarnaast', 'bezigen', 'willen', 'gaan', 'vervolgens', 'raken', 'weten', 'proberen', 'echter', 'vraag', 'verdenken', 'vervatten', 'beslissing', 'hoger_beroep', 'verkort_vonni',
  'geacht', 'instellen', 'ander', 'zien', 'toebehoren', 'hoeveelheid', 'lijst_ii', 'bereiken'])


# ------------------------------------------- Data

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

# owner: shivp Kaggle. Source: https://data.mendeley.com/datasets
# dataset was modified. Original data: https://www.kaggle.com/shivkp/customer-behaviour

df18 = pd.read_csv(DATA_PATH.joinpath('/Users/dj/Python - UvA/DSP/data_2018.csv'), index_col=0) 
df19 = pd.read_csv(DATA_PATH.joinpath('/Users/dj/Python - UvA/DSP/data_2019.csv'), index_col=0) 
df20 = pd.read_csv(DATA_PATH.joinpath('/Users/dj/Python - UvA/DSP/data_2020.csv'), index_col=0) 
df21 = pd.read_csv(DATA_PATH.joinpath('/Users/dj/Python - UvA/DSP/data_2021.csv'), index_col=0) 

convert_dict = {'lemm': str,
                'year': str}
  
df18 = df18.astype(convert_dict)
df19 = df19.astype(convert_dict)
df20 = df20.astype(convert_dict)
df21 = df21.astype(convert_dict)

d = [df18, df19, df20, df21]

for i in d:
    test_list = []
    for j in i['lemm']:
        test_list.append(j.split())
    
    i['lemm_tok'] = test_list

ds = [df18, df19, df20, df21]

all_dats = pd.concat(ds)


big_list = []
for i in ds:
    flat_list = " "
    # iterating over the data
    for item in i['lemm']:
        flat_list = flat_list + " " + item
    big_list.append(flat_list)

tfidf_vectorizer = TfidfVectorizer(input=big_list, stop_words=stop_words)
tfidf_vector = tfidf_vectorizer.fit_transform(big_list)
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=tfidf_vectorizer.get_feature_names())
tfidf_df = tfidf_df.stack().reset_index()
tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term', 'level_2': 'term'})
top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]) # .groupby(['document']).head(5)

years = ('2018', '2019', '2020', '2021')
term = 'minderjarig'

values = top_tfidf['tfidf'][top_tfidf['term'] == term]
terms = list(np.unique(top_tfidf['term']))

# ------------------------------------

layout = html.Div([
    html.H1('TF-IDF Score of a certain word (2018 - 2021)', style={"textAlign": "center"}),
    html.H2('Please select or search for a certain term', style={"textAlign": "left"}),

    html.Div([
        html.Div(dcc.Dropdown(
            id='y2-dropdown', value='hennep', clearable=False,
            options=[{'label': x, 'value': x} for x in terms]
        ), className='six columns'),
    ], className='row'),

    dcc.Graph(id="line-chart"),
])

@app.callback(
    Output("line-chart", "figure"), 
    [Input("y2-dropdown", "value")])
def update_line_chart(term):
    fig = px.line(top_tfidf[top_tfidf['term'] == term],
    x=[2018, 2019, 2020, 2021],
    y="tfidf",
    title='TF-IDF Scores of term: {}'.format(term), height = 700)
    fig.update_xaxes(type='category')
    return fig

