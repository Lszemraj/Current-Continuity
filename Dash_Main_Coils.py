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

#defining camera

camera = dict(
    up = dict(x=0, y=1, z=0),
    center = dict(x=0, y=0, z=0),
    eye = dict(x=-2, y=1.25, z=-1.25)
)



#Dash App Config
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


   html.Div([
    html.Div([
               html.H3('Plot of Entire Selected Solenoid'),
            dcc.Graph(id="Coil1")
                      #style={'width': '90vh', 'height': '90vh'})
            ],className="six columns"),
      ])


])


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
    xs, ys, zs, cs = get_many_thick_cylinders(df_raw.iloc[num_first:num_last])
    cyl.add_traces(data = go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
                     colorscale=[[0, 'blue'], [1, 'gray']],
                     showscale=False,
                     showlegend=True,
                     opacity=1.0,
                     name="Coils", ))
    cyl.update_layout(title=f'{solenoid} Coils',
                      scene = dict(aspectmode = 'data', camera = camera),
                      autosize = False, width = 1600, height = 800
                      )
    return cyl
'''
    for num in range(num_first, num_last):
        x, y, z = get_thick_cylinder_surface_xyz(df_raw, num)
 cyl = go.Figure()
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
    '''


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug= False, port=8050)