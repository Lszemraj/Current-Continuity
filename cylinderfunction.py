import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
#from main.py import *
from scipy.spatial.transform import Rotation



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

def plot_cylinder(r, L, Xc, Yc, Zc, angle):
    X, Y, Z = cylinder(r, L)
    pos = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    data_shape = X.shape
    rot_angles = np.array([0, angle, 0])
    rot = Rotation.from_euler('XYZ', rot_angles, degrees=True)
    pos_rot = rot.apply(pos.T)
    X, Y, Z = pos_rot.T
    X = X.reshape(data_shape)
    Y = Y.reshape(data_shape)
    Z = Z.reshape(data_shape)
    X = X + Xc
    Y = Y + Yc
    Z = Z + Zc
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=[[0, color], [1, color]],opacity = .4,  showscale=False))

print("cylinderfunction.py is set to: {}" .format(__name__))



if __name__ == "__main__":
    df = pd.read_pickle("/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl")
    fig = go.Figure()
    rows = len(df.index)
    color = "blue"
    
    for i in range(0,rows):
        r = df["Ro"].iloc[i]
        L = df["L"].iloc[i]
        Xc =  df['x'].iloc[i]
        Yc = df['y'].iloc[i]
        Zc = df['z'].iloc[i]
        angle =  df['rot1'].iloc[i]
        plot_cylinder(r, L, Xc, Yc, Zc, angle)
        
    fig.update_layout(title='All Coils Plotted except rotated TS ', autosize=False,
                      width=800, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.update_layout(
        scene=dict(
            aspectratio=dict(
                x=10,
                y=2,
                z=30
            ),
            aspectmode='manual'
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=-4,
            dtick=0.75
        )
    )
    fig.show()


