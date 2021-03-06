import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from Plot_update import *
import dash
from dash import Dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html

#defining camera
...
camera = dict(
    up = dict(x=0, y=1, z=0),
    center = dict(x=0, y=0, z=0),
    eye = dict(x=-2, y=1.25, z=-1.25)
)
...
settings_dict = {'coils': {'index_range':[50,None], 'get_func': get_many_thick_cylinders, 'color': 'rgba(138, 207, 103, 0.8)', 'label': 'Coils', 'lgroup': '1'},
                 'straights': {'index_range':[None,None], 'get_func': get_many_3d_straights, 'color': 'rgba(210, 0, 0, 0.8)', 'label': 'Busbars (straight)', 'lgroup': '2'},
                 'arcs': {'index_range':[None,None], 'get_func': get_many_3d_arcs, 'color': 'rgba(210, 0, 0, 0.8)', 'label': 'Busbars (arc)', 'lgroup': '3'},
                 'arcs_transfer': {'index_range':[None,None], 'get_func': get_many_3d_arcs, 'color': 'rgba(210, 0, 0, 0.8)', 'label': None, 'lgroup': '3'},}


#Dash App Config
app = dash.Dash(__name__)
...

app.layout = html.Div([
html.Div(children=[html.Div([
        html.H1(children = 'Current Continuity')])]),
html.Div(
        [
            html.Div(
                [
                    html.H3("""Select Solenoid""",
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
...

#App callback for coil selection

...
@app.callback(
    Output('Coil1', 'figure'),
    Input('solenoid-dropdown', 'value'))

def update_output(input_solenoid):
    df_raw = load_data("Mu2e_Coils_Conductors.pkl")
    df_dict = load_all_geoms(return_dict=True)
    solenoid = input_solenoid

    in_solenoid = df_raw.loc[df_raw['Solenoid'] == solenoid]

    num_first = in_solenoid["Coil_Num"].iloc[0]
    num_last = in_solenoid["Coil_Num"].iloc[-1]

    cyl = go.Figure()
    xs, ys, zs, cs = get_many_thick_cylinders(df_raw.iloc[num_first:num_last])
    cyl.add_traces(data = go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
                     colorscale= [[0, 'rgba(0,0,0,0)'],[1, 'rgba(138, 207, 103, 1)']],
                     showscale=False,
                     showlegend=True,
                     opacity=1.0,
                     name="Coils", ))

#adding bus bars
    if input_solenoid == 'DS':
        data = []
        for k in df_dict.keys():
            s = settings_dict[k]
            df_ = df_dict[k].iloc[s['index_range'][0]:s['index_range'][1]]
            if s['label'] is None:
                sl = False
            else:
                sl = True
            xs, ys, zs, cs = s['get_func'](df_)
            data.append(
                go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
                       colorscale=[[0, 'rgba(0,0,0,0)'], [1, s['color']]],
                       showscale=False,
                       showlegend=sl,
                       legendgroup=s['lgroup'],
                       opacity=1.0,
                       name=s['label'],
                       )
        )
            cyl.add_traces(data =data)
    else:
        pass
    cyl.update_layout(title=f'{solenoid} Coils',
                      scene = dict(aspectmode = 'data', camera = camera),
                      autosize = False, width = 1300, height = 800
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
...

#Run Dash App
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug= False, port=8050)