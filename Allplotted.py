import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

#from main.py import *
from scipy.spatial.transform import Rotation

df = pd.read_pickle("/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl")

color="blue"
def cylinder(r, L, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(-L/2, L/2, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

fig = go.Figure()

DS_coils = pd.DataFrame(df.loc[df['Solenoid'] ==  'DS'])
PS_coils = pd.DataFrame(df.loc[df['Solenoid'] ==  'PS'])
TS_coils = pd.DataFrame(df.loc[df['Solenoid'] ==  'TS'])


for i in range(10):
    r = DS_coils["Ro"].iloc[i]
    L = DS_coils["L"].iloc[i]
    print("r =", r)
    print("L=", L)
    X, Y, Z = cylinder(r, L)
    print(f"x.shape={X.shape}, y.shape {Y.shape}, z.shape = {Z.shape}")
    Xc = DS_coils['x'].iloc[i]
    Yc =DS_coils['y'].iloc[i]
    Zc = DS_coils["z"].iloc[i]
    X = X + Xc
    Y = Y + Yc
    Z = Z + Zc
    fig.add_trace(go.Surface(z=Z, x=X, y=Y,colorscale=[[0,color],[1,color]], showscale = False ))
for i in range(3):
    r = PS_coils["Ro"].iloc[i]
    L = PS_coils["L"].iloc[i]
    print("r =", r)
    print("L=", L)
    X, Y, Z = cylinder(r, L)
    print(f"x.shape={X.shape}, y.shape {Y.shape}, z.shape = {Z.shape}")
    Xc = PS_coils['x'].iloc[i]
    Yc =PS_coils['y'].iloc[i]
    Zc = PS_coils["z"].iloc[i]
    X = X + Xc
    Y = Y + Yc
    Z = Z + Zc
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=[[0,color],[1,color]], showscale = False))
for i in range(52):
    if TS_coils["rot1"].iloc[i] == 0:
        r = TS_coils["Ro"].iloc[i]
        L = TS_coils["L"].iloc[i]
        print("r =", r)
        print("L=", L)
        X, Y, Z = cylinder(r, L)
        print(f"x.shape={X.shape}, y.shape {Y.shape}, z.shape = {Z.shape}")
        Xc = TS_coils['x'].iloc[i]
        Yc =TS_coils['y'].iloc[i]
        Zc = TS_coils["z"].iloc[i]
        X = X + Xc
        Y = Y + Yc
        Z = Z + Zc
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=[[0,color],[1,color]], showscale = False))
                                 #surfacecolor = 'red'))
    if TS_coils["rot1"].iloc[i] != 0:
        pass

        r = TS_coils["Ro"].iloc[i]
        L = TS_coils["L"].iloc[i]
        print("r =", r)
        print("L=", L)
        X, Y, Z = cylinder(r, L)
        print(f"x.shape={X.shape}, y.shape {Y.shape}, z.shape = {Z.shape}")
        #rotate
        #Got stuck here with transpose

        pos = np.array([X.flatten(), Y.flatten(), Z.flatten()])

        print(X.shape)
        print(Y.shape)
        print(Z.shape)
        print(pos)
        data_shape = X.shape

        angle = TS_coils["rot1"].iloc[i]
        rot_angles = np.array([0, angle, 0])
        rot = Rotation.from_euler('XYZ', rot_angles, degrees=True)
        pos_rot = rot.apply(pos.T)

        print("pos_rot shape", pos_rot.shape)
        X, Y, Z = pos_rot.T
        X = X.reshape(data_shape)
        Y = Y.reshape(data_shape)
        Z = Z.reshape(data_shape)

        Xc = TS_coils['x'].iloc[i]
        Yc = TS_coils['y'].iloc[i]
        Zc = TS_coils["z"].iloc[i]
        X = X + Xc
        Y = Y + Yc
        Z = Z + Zc


        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=[[0,color],[1,color]], showscale = False))


fig.update_layout(title='All Coils Plotted except rotated TS ', autosize=False,
                  width=800, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.update_layout(
    scene = dict(
        aspectratio= dict(
            x = 10,
            y = 2,
            z = 30
        ),
        aspectmode = 'manual'
    ),
    xaxis = dict(
        tickmode = 'linear',
        tick0 = -4,
        dtick = 0.75
    )
)


fig.show()

