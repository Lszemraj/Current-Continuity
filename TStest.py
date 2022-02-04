import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial.transform import Rotation
import dash
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

df = pd.read_pickle("/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl")

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

TS_coils = pd.DataFrame(df.loc[df['Solenoid'] ==  'TS'])
fig = go.Figure()
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
        fig.add_trace(go.Surface(z=Z, x=X, y=Y))
    if TS_coils["rot1"].iloc[i] != 0:
        pass
        '''
        r = TS_coils["Ro"].iloc[i]
        L = TS_coils["L"].iloc[i]
        print("r =", r)
        print("L=", L)
        X, Y, Z = cylinder(r, L)
        print(f"x.shape={X.shape}, y.shape {Y.shape}, z.shape = {Z.shape}")
        Xc = TS_coils['x'].iloc[i]
        Yc = TS_coils['y'].iloc[i]
        Zc = TS_coils["z"].iloc[i]
        X = X + Xc
        Y = Y + Yc
        Z = Z + Zc
        #rotate
        #Got stuck here with transpose

        pos = np.array([X, Y, Z]).shape
        print(X.shape)
        print(Y.shape)
        print(Z.shape)
        print(pos)
        angle = TS_coils["rot1"].iloc[i]
        rot_angles = np.array([0, angle, 0])
        rot = Rotation.from_euler('XYZ', rot_angles, degrees=True)
        pos_rot = rot.apply(pos.T)
        fig.add_trace(go.Surface(z=Z, x=X, y=Y))
       '''
        
        


fig.update_layout(title='DS Coils', autosize=False,
                 width=800, height=500,
                 margin=dict(l=65, r=50, b=65, t=90))
fig.show()