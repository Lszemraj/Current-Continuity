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


#geom_df_mu2e = pd.read_pickle('/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl')

df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
dff = df_bars.assign(ending_z =  df_bars['z0'] + df_bars['length'])




app = dash.Dash(__name__)


app.layout = html.Div([
html.Div(children=[html.Div([
        html.H1(children = 'Current Continuity')])]),
html.Div(
        [
            html.Div(
                [
                    html.H6("""Select Solenoid""",
                            style={'margin-right': '2em'})
                ],

            ),
            dcc.Dropdown(
                 id='solenoid-dropdown',
                options=[
            {'label': 'Production Solenoid (PS)', 'value': 'PS'},
            {'label': 'Transport Solenoid (TS)', 'value': 'TS'},
            {'label': 'Detector Solenoid (DS)', 'value': 'DS'},
        ], value = 'DS',

            )
        ],
        #style=dict(display='flex')
    ),

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
    html.Div([
               html.H3('Plot of Entire Selected Solenoid'),
            dcc.Graph(id="Coil1")
                      #style={'width': '90vh', 'height': '90vh'})
            ],className="six columns"),
        html.Div([
            html.H3('Plot of Coils and Longitudinal Bars within Z Range'),
            dcc.Graph(id="Coil2", style={'width': '90vh', 'height': '90vh'})
        ], className="six columns"), ], className="row" ),



])

camera = dict(
    up = dict(x=0, y=1, z=0),
    center = dict(x=0, y=0, z=0),
    eye = dict(x=-2, y=1.25, z=-1.25)
)


@app.callback(
    Output('Coil1', 'figure'),
    Input('solenoid-dropdown', 'value'))

def update_output(input_solenoid):
    df_raw = load_data("Mu2e_Coils_Conductors.pkl")

    solenoid = input_solenoid

    in_solenoid = df_raw.loc[df_raw['Solenoid'] == solenoid]

    num_first = in_solenoid["Coil_Num"].iloc[0]
    num_last = in_solenoid["Coil_Num"].iloc[-1]

    cyl = go.Figure()


    for num in range(num_first, num_last):
        x, y, z = get_thick_cylinder_surface_xyz(df_raw, num)

        cyl.add_traces(data=go.Surface(x=x, y=y, z=z,
                                       surfacecolor = np.ones_like(x),
                                       colorscale = [[0, 'red'], [1, 'red']],
                                       showscale=False,
                                       showlegend=False,
                                       name='Coils (radial center)',
                                       ))

    cyl.update_layout(title=f'{solenoid} Coils',
                      scene = dict(aspectmode = 'data', camera = camera),
                      autosize = False, width = 1600, height = 800
                      )
    return cyl

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
    cyl2.update_layout(title= f'DS Coils and Longitudinal Bars from Z range {start_z_selected} to {end_z_selected}',
                       scene=dict(aspectmode='data', camera=camera),
                       autosize=False, width=1600, height=800
                      )
    return cyl2






app.run_server(debug=True)



