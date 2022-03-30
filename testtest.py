import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from funcs_file import *

import dash
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

df_dict = load_all_geoms(return_dict=True)
generate_dict, conductor_dict = make_conductor_dicts(df_dict)

print(list(conductor_dict.items())[0][0])
print(conductor_dict)