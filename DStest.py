import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
#from main.py import *
from scipy.spatial.transform import Rotation

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



fig = go.Figure()
#fig.update_layout(title='DS Coils', autosize=False,
                #  width=500, height=500,
                 # margin=dict(l=65, r=50, b=65, t=90))
#DS_coils = df.loc[df['Solenoid'] ==  'DS']
DS_coils = pd.DataFrame(df.loc[df['Solenoid'] ==  'DS'])
print(DS_coils)
print(df.columns)
print("this is DS_coils:", DS_coils.columns)
#print(df["Solenoid"][:])
print(DS_coils["L"].iloc[10])
length = len(DS_coils)

range(11)
#for row, index in DS_coils.iterrows():
    #print(DS_coils['Solenoid'].iloc[index])

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
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale = 'teal'))

'''
r = DS_coils["Ro"].iloc[0]
L = DS_coils["L"].iloc[0]
print("r =", r)
print("L=", L)
X, Y, Z = cylinder(r, L)
print(f"x.shape={X.shape}, y.shape {Y.shape}, z.shape = {Z.shape}")
Xc = DS_coils["x"].iloc[0]
Yc = DS_coils["y"].iloc[0]
Zc = DS_coils["z"].iloc[0]
X = X + Xc
Y = Y + Yc
Z = Z + Zc
#fig.add_trace([{type:'mesh3d'}, go.Surface(z=Z, x=X, y=Y)])
'''
#fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(title='DS Coils', autosize=False,
                 width=800, height=500,
                 margin=dict(l=65, r=50, b=65, t=90))
fig.show()

print(df['z'].iloc[-1])


