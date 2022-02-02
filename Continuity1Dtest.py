import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v1.pkl')

#print(df)

segment1 = [df['input1'].iloc[3], df['output1'].iloc[3]]
segment2 = [df['input2'].iloc[0], df['output2'].iloc[0]]

#Check continuity
def check_1d_continuity(segment1, segment2):
    cutoff = 1e-4
    if abs(segment2[0]-segment1[1]) < cutoff:
        continuity = True
    else:
        continuity = False
    return continuity

#print(check_1d_continuity(segment1,segment2))
continuity = []

for i in range(0,200):
    segment1 = [df['input1'].iloc[i], df['output1'].iloc[i]]
    segment2 = [df['input2'].iloc[i], df['output2'].iloc[i]]
    x = check_1d_continuity(segment1, segment2)
    continuity.append(x)

dff =pd.DataFrame(continuity)
dff.columns = ['connected']

#Check that connected determination is same in my calc as in datafile
for i in range(0, 200):
    x = df['connected'].iloc[i]
    y = dff['connected'].iloc[i]
    if x == y:
        same = True
    else:
        same = False
        print(same)


#Plot any two line segments
fig = go.Figure()
row_value = 13
fig.add_trace(
    go.Scatter(
        x= [df['input1'].iloc[row_value], df['output1'].iloc[row_value]],
        y=[0, 0],
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=False))
fig.add_trace(
    go.Scatter(
        x= [df['input1'].iloc[row_value], df['output1'].iloc[row_value]],
        y=[0, 0],
        mode="markers",
        #line=go.scatter.Line(color="gray"),
        showlegend=False))

fig.add_trace(
    go.Scatter(
        x= [df['input2'].iloc[row_value], df['output2'].iloc[row_value]],
        y=[0, 0],
        mode="lines",
        line=go.scatter.Line(color="blue"),
        showlegend=False))

fig.add_trace(
    go.Scatter(
        x= [df['input2'].iloc[row_value], df['output2'].iloc[row_value]],
        y=[0, 0],
        mode="markers",
        #line=go.scatter.Line(color="gray"),
        showlegend=False))
fig.update_layout(title="Continuity Test 1D")
fig.show()