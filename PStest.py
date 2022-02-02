import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

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

PS_coils = pd.DataFrame(df.loc[df['Solenoid'] ==  'PS'])
fig = go.Figure()
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
    fig.add_trace(go.Surface(z=Z, x=X, y=Y))


fig.update_layout(title='DS Coils', autosize=False,
                 width=800, height=500,
                 margin=dict(l=65, r=50, b=65, t=90))
fig.show()