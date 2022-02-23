import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from Plot_update import *

import dash
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

df_dict = load_all_geoms(return_dict=True)

df_ = df_dict['arcs']
print(df_.columns)