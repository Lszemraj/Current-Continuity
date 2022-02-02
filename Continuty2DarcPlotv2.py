import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from plotly import __version__

import plotly.graph_objects as go

fig = go.Figure()

# Set axes ranges
fig.update_xaxes(range=[0, 7])
fig.update_yaxes(range=[0, 2.5])


df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/arc_segments_2d_v3.pkl')
i =2

Cx = df['xc'].iloc[i]
Cy = df['yc'].iloc[i]
r = df['R'].iloc[i]
angle = df['phi0'].iloc[i]
delta_angle = df['dphi'].iloc[i]
X = Cx + (r * np.cos(angle))
Y = Cy + (r * np.sin(angle))
X_final = Cx + (r * np.cos(angle + delta_angle))
Y_final = Cy + (r * np.sin(angle + delta_angle))

fig.add_shape(type="circle",
    x0 = Cx - r, y0= Cy - r, x1 = Cx + r, y1 = Cy +r,  #x1=1, y1=2,
    line=dict(color="RoyalBlue",width=3)
)



fig.add_trace(go.Scatter(
    x = [X],
    y = [Y],
    mode = 'markers',
    #size = 10
)
)
fig.add_trace(go.Scatter(
    x = [X_final],
    y = [Y_final],
    mode = 'markers',
    #size = 10
)
)

fig.update_layout(
    margin=dict(l=20, r=20, b=100),
    height=800, width=800,
    #plot_bgcolor="white"
)

fig.show()

print("radius is:", r)
#red is start, blue is end angle