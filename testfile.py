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
#print(df_raw.columns)
in_solenoid = df_raw.loc[df_raw['Solenoid'] == solenoid]
#print(in_solenoid)
#print(df_raw['z'].iloc[55], df_raw['z'].iloc[65])
num_first = in_solenoid["Coil_Num"].iloc[0]
num_last = in_solenoid["Coil_Num"].iloc[-1]
#print(num_first, num_last)

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
#print(df_bars)


#print(df_bars.columns)

z_start = df_bars['y0'].iloc[0]
z_end =  df_bars['z0'].iloc[2] + df_bars['length'].iloc[2]
z_values = np.arange(z_start, z_end)

num = len(z_values)
x_values = [df_bars['x0'].iloc[2]]*num

y_values = [df_bars['y0'].iloc[2]]*num
#print(df_bars['z0'].iloc[0], df_bars['z0'].iloc[-1])
#print(df_bars['length'][:])
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
#print(dff)

end_z_selected = 13.64073
start_z_selected = 3.74888
coils = df_raw.query(f'z < {end_z_selected} and z >= {start_z_selected}')
bars = dff.query(f'z0 < {end_z_selected} and z0 >= {start_z_selected}')
#print(coils)
#print(bars)
index = bars.index
#print(index)
idx = index.tolist()
#print(idx)
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

print(df_bars)
print(df_bars.columns)
print(df_bars['x0'].iloc[1])
i=15
x0 = df_bars['x0'].iloc[i]
y0 = df_bars['y0'].iloc[i]
z0 = df_bars['z0'].iloc[i]
zf = x0 = df_bars['z0'].iloc[i] + df_bars['length'].iloc[i]
width =  df_bars['W'].iloc[i]
W2 = width/2
print("width", width)

xc = [x0 + W2, x0 + W2, x0-W2, x0-W2, x0 + W2, x0 + W2, x0-W2, x0-W2]
yc = [y0 + W2, y0 + W2, y0-W2, y0-W2, y0 + W2, y0 + W2, y0-W2, y0-W2]
zc = [z0, z0, z0, z0, zf, zf, zf, zf]
'''
fig = go.Figure(data=[
     go.Scatter3d(x=x, y=y, z=z,
                  mode='markers',
                  marker=dict(size=2)
                 ),
     go.Mesh3d(
        # 8 vertices of a cube
        x=xc,
        y=yc,
        z=zc,

        alphahull = 0,
        opacity=0.6,
        color='#DC143C',
        flatshading = True
    )
    ])
'''

df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
i=0

def create_bar_endpoints(df_bars, index):
    x0 = df_bars['x0'].iloc[i]
    y0 = df_bars['y0'].iloc[i]
    z0 = df_bars['z0'].iloc[i]
    length = df_bars['length'].iloc[i]
    zf = x0 = df_bars['z0'].iloc[i] + length
    width = df_bars['W'].iloc[i]
    Thickness = df_bars['T'].iloc[i]
    T2 = Thickness / 2
    W2 = width / 2
    xc = [x0 + W2, x0 + W2, x0 - W2, x0 - W2, x0 + W2, x0 + W2, x0 - W2, x0 - W2]
    yc = [y0 + T2, y0 - T2, y0 - T2, y0 + T2, y0 + T2, y0 - T2, y0 - T2, y0 + T2]
    zc = [z0, z0, z0, z0, zf, zf, zf, zf]
    return xc, yc, zc



x0 = df_bars['x0'].iloc[i]
y0 = df_bars['y0'].iloc[i]
z0 = df_bars['z0'].iloc[i]
zf = x0 = df_bars['z0'].iloc[i] + df_bars['length'].iloc[i]
width =  df_bars['W'].iloc[i]
Thickness = df_bars['T'].iloc[i]
T2 = Thickness/2
W2 = width/2
print("width", width)
length = df_bars['length'].iloc[i]

#xc = [x0 + W2, x0 + W2, x0 - W2, x0 - W2, x0 + W2, x0 + W2, x0 - W2, x0 - W2]
#yc = [y0 + T2, y0 - T2, y0 - T2, y0 + T2, y0 + T2, y0 - T2, y0 - T2, y0 + T2]
#zc = [z0, z0, z0, z0, zf, zf, zf, zf]
xc = [W2,  W2, -W2, -W2, W2, W2, -W2, -W2]
yc = [T2, -T2, -T2, T2, T2, -T2, -T2, T2]
zc = [-length/2 , -length/2, -length/2, -length/2, length/2, length/2, length/2, length/2]

pos = np.array([xc, yc, zc])
phi2 = df_bars["Phi2"].iloc[i]
theta2 = df_bars["theta2"].iloc[i]
psi2 = df_bars["psi2"].iloc[i]
rot_angles = np.array([phi2, theta2, psi2])
rot = Rotation.from_euler('ZYZ', rot_angles, degrees=True)
pos_rot = rot.apply(pos.T)
X, Y, Z = pos_rot.T

X = X + x0
Y = Y + y0
Z = Z + z0

print(xc,yc,zc)
print(X,Y,Z)

fig = go.Figure(data = # go.Scatter3d(x=xc,y=yc,z=zc))
                go.Mesh3d(x=X,y=Y,z=Z, alphahull = 0, intensity = np.linspace(1, 1, 8, endpoint=True),name='y'))
#go.Mesh3d(x=xc,y=yc,z=zc, alphahull = 0, intensity = np.linspace(1, 1, 8, endpoint=True),name='y')])

fig.update_layout(scene = dict(aspectmode = 'data'))
fig.show()

#Considering angle rotations

df_raw = load_data("Mu2e_Coils_Conductors.pkl")
print(df_raw)
print(df_raw.columns)