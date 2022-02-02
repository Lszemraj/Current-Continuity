import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial.transform import Rotation

df = pd.read_pickle("/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl")

#x = np.linspace(0,100)
#y = np.linspace(0,100)
#X, Y = np.meshgrid(x,y)
#print(X, Y)

#fig = go.Figure(data=[go.Surface(x=X, y=Y, z= )])

#def linear_circular_arc():

def cylinder(r, L, end_angle, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    angle = end_angle*(np.pi/180)
    theta = np.linspace(0, angle, nt)
    v = np.linspace(-L/2, L/2, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def plot_cylinder(r, L, Xc, Yc, Zc,end_angle, angle):
    X, Y, Z = cylinder(r, L, end_angle)
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
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=[[0, color], [1, color]], showscale=False))


if __name__ == "__main__":
    df = pd.read_pickle("/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl")
    fig = go.Figure()
    rows = len(df.index)
    color = "blue"
    end_angle = 87.4
    r = 1.18
    L =  0.63
    Xc = -4.924
    Yc = 0.589
    Zc = 8.558
    angle = 0
    plot_cylinder(r, L, Xc, Yc, Zc, end_angle, angle)
    fig.show()

