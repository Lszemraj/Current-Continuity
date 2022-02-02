import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v4.pkl')
print(df.head)

def check_1d_overlap(segment1, segment2):
    cutoff = 1e-4
    if abs(segment2[0] - segment1[1]) < cutoff:
        continuity = True
    else:
        continuity = False
    if segment2[0] - segment1[1] < 0:
        overlapped = True
    else: #if segment2[0] - segment1[1] >= 0:
        overlapped = False
    return continuity, overlapped

length = len(df)
continuity = []
overlap = []

for i in range(0,length):
    if i ==  0:
        x = True
        y = False
        continuity.append(x)
        overlap.append(y)
    else:
        segment1 = [df['input'].iloc[i-1], df['output'].iloc[i-1]]
        segment2 = [df['input'].iloc[i], df['output'].iloc[i]]
        x,y = check_1d_overlap(segment1, segment2)
        continuity.append(x)
        overlap.append(y)

data = {'connected_to_previous': continuity, 'overlapped_with_previous': overlap}
testv4df = pd.DataFrame(data)
print(testv4df)






#check if I am correct
overlapped_with_previous =[]
connected_to_previous = []
for i in range(0, length-1):
    x = df['connected_to_previous'].iloc[i]
    y = testv4df['connected_to_previous'].iloc[i]
    x1 = df['overlapped_with_previous'].iloc[i]
    y1 = testv4df['overlapped_with_previous'].iloc[i]
    if x == y:
        same = 'CORRECT'
    else:
        same = 'NOT SAME'
    connected_to_previous.append(same)
    if x1 == y1:
        same1 = 'CORRECT'
    else:
        same1 = 'NOT SAME'
    overlapped_with_previous.append(same1)
data1 = {'connected_checked': connected_to_previous, 'overlapped_checked':overlapped_with_previous}
checked = pd.DataFrame(data1)

print(checked.loc[checked['connected_checked'] == 'NOT SAME'])
print(checked.loc[checked['overlapped_checked'] == 'NOT SAME'])

