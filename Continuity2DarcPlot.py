import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from plotly import __version__




import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

fig = go.Figure()
df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/arc_segments_2d_v2.pkl')
'''
def find_2D_arcpoints(Cx, Cy, r, angle, delta_angle):
    X_final = Cx + (r * np.cos(angle + delta_angle))
    Y_final = Cy + (r * np.sin(angle + delta_angle))
    return  X_final, Y_final

X_points = []
Y_points = []

def plot_circle(df, i):
    start = df["phi0"]
    delta_angle = df['dphi'].iloc[i]
    #array = np.arange(start = start, stop = delta_angle)
    array = np.linspace(start = start, stop = delta_angle, num = 20, dtype= int).reshape(1, 20)
    data = {"angles": array}
    print(data)
    dff = pd.DataFrame(data)
    Cx = df['xc'].iloc[i]
    Cy = df['yc'].iloc[i]
    r = df['R'].iloc[i]
    X_initial = Cx + (r * np.cos(start))
    X_points.append(X_initial)
    Y_initial = Cy + (r * np.sin(start))
    Y_points.append(Y_initial)
    for n in range(0, len(dff)):
        if dff["angles"].iloc[n] == start:
            pass
        else:
            X, Y = find_2D_arcpoints(Cx, Cy, r, start, dff["angles"].iloc[n])
            X_points.append(X)
            Y_points.append(Y)
    for x_point, y_point in X_points, Y_points:
        fig.add_trace(go.Scatter(
            x= x_point,
            y= y_point,
            #text=["Unfilled Circle"],
            #mode="text",
        ))


plot_circle(df, 1)
fig.show()


'''


'''
fig.add_trace(go.Scatter(
    x=[1.5, 3.5],
    y=[0.75, 2.5],
    text=["Unfilled Circle"],
    mode="text",
    ))

fig.update_xaxes(range=[0, 4.5], zeroline=False)
fig.update_yaxes(range=[0, 4.5])

# Add circles
fig.add_shape(type="circle",
                xref="x", yref="y",
                x0=1, y0=1, x1=3, y1=3,
                line_color="LightSeaGreen",)

fig.update_layout(width=800, height=800)
'''


def my_circle(center, radius, n_points=75):
    t = np.linspace(0, 1, n_points)
    x = center[0] + radius * np.cos(2 * np.pi * t)
    y = center[1] + radius * np.sin(2 * np.pi * t)
    return x, y


trace = dict(x=[2.5, 4], y=[1.75, 2.35],
             mode='markers',
             marker=dict(size=9, color='rgb(44, 160, 101)'))

axis = dict(showline=True, zeroline=False, showgrid=False)




i =1

center = [df['xc'].iloc[i], df['yc'].iloc[i]]
r = df['R'].iloc[i]
x, y = my_circle(center, r)

path = 'M ' + str(x[0]) + ',' + str(y[1])
for k in range(1, x.shape[0]):
    path += ' L ' + str(x[k]) + ',' + str(y[k])
path += ' Z'

layout = dict(width=450, height=450, autosize=False,
              xaxis=dict(axis, **dict(range=[1, 6])),
              yaxis=dict(axis, **dict(range=[-1, 4])),
              shapes=[dict(type='path',
                           layer='below',
                           path=path,
                           fillcolor='rgba(44, 160, 101, 0.5)',
                           line=dict(color='rgb(44, 160, 101)')

                           )]
              )
fig = dict(data=[trace], layout=layout)
iplot(fig)

