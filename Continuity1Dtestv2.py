import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v2.pkl')

#Continuity test with line of values and a break

print(df)

def check_1d_continuity(segment1, segment2):
    cutoff = 1e-4
    if abs(segment2[0] - segment1[1]) <= cutoff:
        continuity = True
    else:
        continuity = False
    return continuity

length = len(df)
continuity = []

#Check Continuity in line
for i in range(0,length):
    if i ==  0:
        x = True
        continuity.append(x)
    else:
        segment1 = [df['input'].iloc[i-1], df['output'].iloc[i-1]]
        segment2 = [df['input'].iloc[i], df['output'].iloc[i]]
        x = check_1d_continuity(segment1, segment2)
        continuity.append(x)


dff =pd.DataFrame(continuity)
dff.columns = ['connected']
#print(dff)





#Compare values to correct connections
connected_to_previous = []
for i in range(0, length-1):
    x = df['connected_to_previous'].iloc[i]
    y = dff['connected'].iloc[i]
    if x == y:
        same = True
    else:
        same = 'NOT SAME'
    connected_to_previous.append(same)

dfff= pd.DataFrame(connected_to_previous)
dfff.columns = ['continuity_same']
#print(dfff)
error_rows = []
for i in range(0,199):
    if dfff['continuity_same'].iloc[i] == 'NOT SAME':
        print(i)
        error_rows.append(i)

#print("number of errors", len(error_rows))





#color
color = []
n =200
for i in range(200):
    if i%2 == 0:
        colorx = "red"
    else:
        colorx = "blue"
    color.append(colorx)









fig = go.Figure()
color = ["red", "blue"]
for i in range(0, length):
    #c = next(color)
    fig.add_trace(
        go.Scatter(
            x= [df['input'].iloc[i], df['output'].iloc[i]],
            y=[0, 0],
            mode="lines",
            line=go.scatter.Line(color= "grey"),
            #color_discrete_sequence=["red", "green"],
            showlegend=False))
    fig.add_trace(
        go.Scatter(
            x= [df['input'].iloc[i], df['output'].iloc[i]],
            y=[0, 0],
            mode="markers",
            #line=go.scatter.Line(color="gray"),
            showlegend=False))

print("REAL DF")
for i in range(0, length):
    if df['connected_to_previous'].iloc[i] == False:
        print(i)
fig.update_layout(title="Continuity Test1DV2")
fig.show()
