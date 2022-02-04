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

input_solenoid = 'DS'

df_raw = load_data("Mu2e_Coils_Conductors.pkl")
solenoid = input_solenoid
print(df_raw.columns)
in_solenoid = df_raw.loc[df_raw['Solenoid'] == solenoid]
print(in_solenoid)
print(df_raw['z'].iloc[55], df_raw['z'].iloc[65])
num_first = in_solenoid["Coil_Num"].iloc[0]
num_last = in_solenoid["Coil_Num"].iloc[-1]
print(num_first, num_last)

cyl = go.Figure()

for num in range(num_first, num_last):
    x, y, z = get_thick_cylinder_surface_xyz(df_raw, num)

    cyl.add_traces(data = go.Surface(x=x, y=y, z=z,
                 showscale =False,
                 showlegend = False,
                 name = 'Coils (radial center)',
                                     colorscale= [[0, 'green'], [0, 'green'], [0, 'green']]
                 ))


cyl.update_layout(title=f'{solenoid} Coils', autosize=True
                  )
#cyl.show()

df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
print(df_bars)


print(df_bars.columns)

z_start = df_bars['y0'].iloc[0]
z_end =  df_bars['z0'].iloc[2] + df_bars['length'].iloc[2]
z_values = np.arange(z_start, z_end)

num = len(z_values)
x_values = [df_bars['x0'].iloc[2]]*num

y_values = [df_bars['y0'].iloc[2]]*num
print(df_bars['z0'].iloc[0], df_bars['z0'].iloc[-1])
print(df_bars['length'][:])
bars_fig = go.Figure(data=go.Scatter3d(
    x= x_values, y= y_values, z=z_values,
    marker=dict(
        size=4,
        color= 'red',
        #colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=2
    )
))

#bars_fig.show()

df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
dff = df_bars.assign(ending_z =  df_bars['z0'] + df_bars['length'])
print(dff)

end_z_selected = 13.64073
start_z_selected = 3.74888
coils = df_raw.query(f'z < {end_z_selected} and z >= {start_z_selected}')
bars = dff.query(f'z0 < {end_z_selected} and z0 >= {start_z_selected}')
print(coils)
print(bars)
index = bars.index
print(index)
idx = index.tolist()
print(idx)
cyl2 = go.Figure()

for i in idx:
    z_start = df_bars['y0'].iloc[i]
    z_end = df_bars['z0'].iloc[i] + df_bars['length'].iloc[i]
    z_values = np.arange(z_start, z_end)

    num = len(z_values)
    x_values = [df_bars['x0'].iloc[i]] * num
    y_values = [df_bars['y0'].iloc[i]] * num

    cyl2.add_traces(data=go.Scatter3d(
        x=x_values, y=y_values, z=z_values,
        marker=dict(
            size=4,
            color='red',
            # colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    ))