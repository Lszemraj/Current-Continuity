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
from dash import dash_table
datadir = '/home/shared_data/helicalc_params/'
df_bars = pd.read_csv(datadir + "Mu2e_Longitudinal_Bars_V13.csv")
#dff = df_bars.assign(ending_z =  df_bars['z0'] + df_bars['length'])
df_raw = load_data("Mu2e_Coils_Conductors.pkl")
L = ['DS-' + str(i) for i in np.arange(1,12)]

df_coils = (df_raw[[ 'Coil_Num', 'L', 'x', 'y', 'z']].iloc[55:]).round(6)
df_coils = df_coils.assign(Assigned_Number = L )
df_coils = df_coils[[ 'Assigned_Number','Coil_Num', 'L', 'x', 'y', 'z']]
units = ['', '', '(m)', '(m)', '(m)', '(m)']




#Camera Angle
...
camera = dict(
    up = dict(x=0, y=1, z=0),
    center = dict(x=0, y=0, z=0),
    eye = dict(x=-2, y=1.25, z=-1.25)
)
...
d_columns = [
    {"name": "Coil_Num",
     "id": "Coil Num"}
]

#Dash App config
app = dash.Dash(__name__)
...

app.layout = html.Div([

html.Div(children=[html.Div([
        html.H1(children = 'Current Continuity')])]),

html.H2(children = 'Datatable of Detector Solenoid Coils'),
        dash_table.DataTable(data = df_coils.to_dict('records'),
                             columns = [{"name": i + j, "id": i} for i, j in zip(df_coils.columns, units)]),


html.Div(
    [
        html.H3("Input starting and ending Z to plot [must be within 3.74888 and 13.7] "),
        html.Br(),
        dcc.Input(id="input1", type="number", placeholder="", style={'marginRight':'10px'}, value = 3.7 ),
        dcc.Input(id="input2", type="number", placeholder="", value = 13.7, debounce=True),
        html.Div(id="output"),
    ]
),
        html.Div([
            html.H2('Plot of Coils and Longitudinal Bars within Z Range'),
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
    df_dict = load_all_geoms(return_dict=True)

    coils = df_raw.query(f'(z < {end_z_selected}) and (z >= {start_z_selected})')
    #bars = df_bars.query(f'(z0 < {end_z_selected}) and (z0 >= {start_z_selected})')
    num_first = coils["Coil_Num"].iloc[0]
    num_last = coils["Coil_Num"].iloc[-1]

    cyl2 = go.Figure()
    #index = bars.index
    #idx = index.tolist()
    #print(bars)
    #print(idx)
    #print(type(idx[0]))
    #print(bars.columns)
    for num in range(num_first, num_last+1):
        x, y, z = get_thick_cylinder_surface_xyz(coils, num)

        cyl2.add_traces(data=go.Surface(x=x, y=y, z=z,
                                        surfacecolor=np.ones_like(x),
                                       colorscale=[[0, 'rgba(0,0,0,0)'],[1, 'rgba(100, 107, 303,1)']],
                                       showscale=False,
                                       showlegend=False,
                                       name='Coils (radial center)',
                                       ))
    #for i in idx:
        #xc, yc, zc = create_bar_endpoints(df_bars, i)
        #cond = df_bars['cond N'].iloc[i]


    data = []
    # straight bars
    df_ = df_dict['straights']
    df_ = df_.assign(zf = df_['z0'] + df_['length'])
    df_str = df_.query(f'(z0 < {end_z_selected}) and (z0 >= {start_z_selected}) and (zf < {end_z_selected}) ')
    for i, num in enumerate(df_str['cond N'].values):
        if i == 0:
            name = 'straight bus bars'
            showlegend = True
        else:
            name = None
            showlegend = False
        xs, ys, zs, cs = get_3d_straight_surface(df_str, num)
        data.append(go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
                                   #                            colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(255, 92, 92, 0.8)']],
                                   colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'pink']],
                                   showscale=False,
                                   showlegend=showlegend,
                                   legendgroup='1',
                                   opacity=1.0,
                                   name=name,
                                   # may see some improvement by changing lighting conditions
                                   lighting=dict(ambient=1.0, roughness=0.0, diffuse=1.0, fresnel=5, specular=2),
                                   )
                        )
    # arc bars
    df_ = df_dict['arcs']
    df_arcs = df_.query(f'(z0 < {end_z_selected}) and (z0 >= {start_z_selected})')
    for i, num in enumerate(df_arcs['cond N'].values):
        if i == 0:
            name = 'arc bus bars'
            showlegend = True
        else:
            name = None
            showlegend = False
        xs, ys, zs, cs = get_3d_arc_surface(df_arcs, num)
        data.append(go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
                                   #                            colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(255, 92, 92, 0.8)']],
                                   colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'pink']],
                                   showscale=False,
                                   showlegend=showlegend,
                                   legendgroup='2',
                                   opacity=1.0,
                                   name=name,
                                   # may see some improvement by changing lighting conditions
                                   lighting=dict(ambient=1.0, roughness=0.0, diffuse=1.0, fresnel=5, specular=2),
                                   )
                        )
    # transfer line arcs
    df_ = df_dict['arcs_transfer']
    df_arcst = df_.query(f'(z0 < {end_z_selected}) and (z0 >= {start_z_selected})')
    for i, num in enumerate(df_arcst['cond N'].values):
        name = None
        showlegend = False
        xs, ys, zs, cs = get_3d_arc_surface(df_arcst, num)
        data.append(go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
                                   #                            colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(255, 92, 92, 0.8)']],
                                   colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'pink']],
                                   showscale=False,
                                   showlegend=showlegend,
                                   legendgroup='2',
                                   opacity=1.0,
                                   name=name,
                                   # may see some improvement by changing lighting conditions
                                   lighting=dict(ambient=1.0, roughness=0.0, diffuse=1.0, fresnel=5, specular=2),
                                   )
                        )
    cyl2.add_traces(data = data)




    cyl2.update_layout(title= f'DS Coils and Bus Bars from Z range {start_z_selected} to {end_z_selected}',
                       legend_title_text='Conductor Number',
                       showlegend = True,
                       scene=dict(aspectmode='data', camera=camera),
                       autosize=False, width=1300, height=800
                      )
    return cyl2
...

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug= True, port=8070)

'''
html.Div([     #html.H3(children = 'Hall Probe Status Datatable'),
        dash_table.DataTable(
        id='table',
        #data= df_raw.to_dict("records"),
            columns = [{"name": i, "id": i, "type": 'numeric'} for i in df_raw.columns],
            cell_selectable = False,
'''