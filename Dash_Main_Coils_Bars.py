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
from dash import dash_table

df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
dff = df_bars.assign(ending_z =  df_bars['z0'] + df_bars['length'])
df_raw = load_data("Mu2e_Coils_Conductors.pkl")
#Camera Angle
...
camera = dict(
    up = dict(x=0, y=1, z=0),
    center = dict(x=0, y=0, z=0),
    eye = dict(x=-2, y=1.25, z=-1.25)
)
...


#Dash App config
app = dash.Dash(__name__)
...

app.layout = html.Div([
html.Div(children=[html.Div([
        html.H1(children = 'Current Continuity')])]),

html.Div([     #html.H3(children = 'Hall Probe Status Datatable'),
        dash_table.DataTable(
        id='table',
        #data= data,
        columns=[{"name": i, "id": i, "type": 'numeric'} for i in df_raw.columns],
        sort_action='native',
        editable=True,),
            '''
        style_data_conditional=[
             {
                 'if': {
                     'column_id': 'Probe_Name',
                     'filter_query' : "{Probe_Name} eq 'SP1'",

                 },
                 'backgroundColor': 'green',
                 'color': 'white'
             },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'SP2'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'SP3'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'BP1'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'BP2'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'BP3'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'BP4'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            },
            {
                'if': {
                    'column_id': 'Probe_Name',
                    'filter_query': "{Probe_Name} eq 'BP5'",

                },
                'backgroundColor': 'green',
                'color': 'white'
            }
        ]

        ),]
        '''
        ]),
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
...


#Callback for Z input

...
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
                                        colorscale=[[0, 'blue'], [1, 'blue']],
                                       showscale=False,
                                       showlegend=False,
                                       name='Coils (radial center)',
                                       ))
    for i in idx:
        xc, yc, zc = create_bar_endpoints(df_bars, i)
        cond = df_bars['cond N'].iloc[i]
        cyl2.add_traces(
            data =  go.Mesh3d(x=xc,y=yc,z=zc, alphahull = 0, intensity = np.linspace(1, 1, 8, endpoint=True),name='y'))

        '''
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
        '''
    cyl2.update_layout(title= f'DS Coils and Longitudinal Bars from Z range {start_z_selected} to {end_z_selected}',
                       legend_title_text='Conductor Number',
                       scene=dict(aspectmode='data', camera=camera),
                       autosize=False, width=1500, height=800
                      )
    return cyl2
...

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug= False, port=8070)