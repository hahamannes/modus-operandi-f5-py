
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
import altair as alt


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


terms = ['2018', '2019', '2020', '2021']

# ------------------------------------------- Data


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()



layout = html.Div([
    html.H1('LDA Visualisation Tool', style={"textAlign": "center"}),
    html.H5('Please select a year', style={"textAlign": "left"}),
    html.Div(dcc.Dropdown(
            id='genre-dropdown', value='2018', clearable=False,
            options=[{'label': x, 'value': x} for x in terms],
        ), className='six columns'),
        html.Iframe(src=app.get_asset_url('lda18.html'),
                                            style=dict(position="absolute", left="50", top="100", width="100%", height="100%"))
])


