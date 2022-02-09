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

df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
dff = df_bars.assign(ending_z =  df_bars['z0'] + df_bars['length'])

#Camera Angle
camera = dict(
    up = dict(x=0, y=1, z=0),
    center = dict(x=0, y=0, z=0),
    eye = dict(x=-2, y=1.25, z=-1.25)
)



#Setting up dash app
app = dash.Dash(__name__)


app.layout = html.Div([
html.Div(children=[html.Div([
        html.H1(children = 'Current Continuity')])]),

html.Div(
    [
        html.I("Input starting and ending Z to plot [must be within 3.74888 and 13.64073] "),
        html.Br(),
        dcc.Input(id="input1", type="number", placeholder="", style={'marginRight':'10px'}, value = 3.74888 ),
        dcc.Input(id="input2", type="number", placeholder="", value = 13.64073, debounce=True),
        html.Div(id="output"),
    ]
),

        html.Div([
            html.H3('Plot of Coils and Longitudinal Bars within Z Range'),
            dcc.Graph(id="Coil2", style={'width': '90vh', 'height': '90vh'})
        ])

])



@app.callback(
    Output('Coil2', 'figure'),
    [Input('input1', 'value'), Input('input2', 'value')])

def update_coils(start_z_selected, end_z_selected):
    df_raw = load_data("Mu2e_Coils_Conductors.pkl")

    coils = df_raw.query(f'z < {end_z_selected} and z >= {start_z_selected}')
    bars = dff.query(f'z0 < {end_z_selected} and z0 >= {start_z_selected}')
    num_first = coils["Coil_Num"].iloc[0]
    num_last = coils["Coil_Num"].iloc[-1]

    cyl2 = go.Figure()
    index = bars.index
    idx = index.tolist()

    for num in range(num_first, num_last):
        x, y, z = get_thick_cylinder_surface_xyz(df_raw, num)

        cyl2.add_traces(data=go.Surface(x=x, y=y, z=z,
                                        surfacecolor=np.ones_like(x),
                                        colorscale=[[0, 'red'], [1, 'red']],
                                       showscale=False,
                                       showlegend=False,
                                       name='Coils (radial center)',
                                       ))
    for i in idx:
        z_start = df_bars['z0'].iloc[i]
        z_end = df_bars['z0'].iloc[i] + df_bars['length'].iloc[i]
        z_values = np.arange(z_start, z_end)

        num = len(z_values)
        x_values = [df_bars['x0'].iloc[i]] * num
        y_values = [df_bars['y0'].iloc[i]] * num
        cond = df_bars['cond N'].iloc[i]
        cyl2.add_traces(data=go.Scatter3d(
            x=x_values, y=y_values, z=z_values,
            marker=dict(
                size=4,
                color='green',
                    # colorscale='Viridis',
            ),
            line=dict(
                color='darkblue',
                width=2
            ),
            name = f'{cond}',
        ))
    cyl2.update_layout(title= f'DS Coils and Longitudinal Bars from Z range {start_z_selected} to {end_z_selected}',
                       legend_title_text='Conductor Number',
                       scene=dict(aspectmode='data', camera=camera),
                       autosize=False, width=1500, height=800
                      )
    return cyl2


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug= False, port=8070)