
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import dash
import sqlalchemy
import pandas as pd
import pathlib
from app import app
import io


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

import altair as alt
import numpy as np

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

# Heatmap 

tfidf18 = pd.read_csv('/Users/dj/Python - UvA/DSP/topidfs_2018.csv', index_col=0) 
tfidf19 = pd.read_csv('/Users/dj/Python - UvA/DSP/topidfs_2019.csv', index_col=0) 
tfidf20 = pd.read_csv('/Users/dj/Python - UvA/DSP/topidfs_2020.csv', index_col=0) 
tfidf21 = pd.read_csv('/Users/dj/Python - UvA/DSP/topidfs_2021.csv', index_col=0) 

top_idfs = [tfidf18, tfidf19, tfidf20, tfidf21]

# TF-IDF Line Plot

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

# ------------------------------- 


layout = html.Div([
    html.H1('TF-IDF Heatmap of Documents & Terms', style={"textAlign": "center"}),
    html.H2('Please select or search for a certain term', style={"textAlign": "left"}),

    html.Div([
        html.Div(dcc.Dropdown(
            id='y2-dropdown', value='hennep', clearable=False,
            options=[{'label': x, 'value': x} for x in terms]
        ), className='six columns'),
    ], className='row'),

        html.Iframe(
        id='plot',
        height = '1000',
        width = '1500',
        sandbox='allow-scripts',
    ),
])

@app.callback(
    dash.dependencies.Output("plot", "srcDoc"), 
    [dash.dependencies.Input("y2-dropdown", "value")])


def update_line_chart(term):
    
    a = 0

    brush = alt.selection_interval()

    # Terms in this list will get a red dot in the visualization
    term_list = [term] # Highlight the words of interest

    # adding a little randomness to break ties in term ranking
    top_tfidf_plusRand = top_idfs[a].copy()
    top_tfidf_plusRand = top_tfidf_plusRand.iloc[:500,]
    top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(top_tfidf_plusRand.shape[0])*0.0001

    # base for all visualizations, with rank calculation
    base = alt.Chart(top_tfidf_plusRand).encode(
        x = 'rank:O',
        y = 'document:N'
    ).transform_window(
        rank = "rank()",
        sort = [alt.SortField("tfidf", order="descending")],
        groupby = ["document"],
    )

    # heatmap specification
    heatmap = base.mark_rect().encode(
        color = 'tfidf:Q'
    )

    # red circle over terms in above list
    circle = base.mark_circle(size=100).encode(
        color = alt.condition(
            alt.FieldOneOfPredicate(field='term', oneOf=term_list),
            alt.value('red'),
            alt.value('#FFFFFF00')        
        )
    )

    # text labels, white for darker heatmap colors
    text = base.mark_text(baseline='middle').encode(
        text = 'term:N',
        color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
    )
    
    # display the three superimposed visualizations
    fig  = (heatmap + circle + text).properties(width = 1200)
    fig = alt.hconcat(fig)

    fig_html = io.StringIO()
    fig.save(fig_html, 'html')

    return fig_html.getvalue()



