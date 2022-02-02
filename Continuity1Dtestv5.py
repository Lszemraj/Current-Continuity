import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v5.pkl')
print(df.head)

#multiple connections

def check_same_input(segment1, segment2):
    cutoff = 1e-4
    if abs(segment1[0] - segment2[0])<= cutoff:
        same_input = True
    else:
        same_input = False
    return same_input

def check_1d_overlap(segment1, segment2):
    cutoff = 1e-4
    if segment2[0] - segment1[1] < cutoff:
        continuity = True
    else:
        continuity = False
    if segment2[0] - segment1[1] < 0:
        overlapped = True
        continuity = False
    else: #if segment2[0] - segment1[1] >= 0:
        overlapped = False
    return continuity, overlapped

same_input1 = []
overlap = []
length = len(df)

for i in range(0,length):
    if i ==  0:
        same_input = False
        same_input1.append(same_input)
        y = False
       # overlap.append(y)
    else:
        segment1 = [df['input'].iloc[i-1], df['output'].iloc[i-1]]
        segment2 = [df['input'].iloc[i], df['output'].iloc[i]]
        same_input = check_same_input(segment1, segment2)
        same_input1.append(same_input)
       # x,y = check_1d_overlap(segment1, segment2)
       # overlap.append(y)

data = {'same_input_as_previous': same_input1}#, "overlapped_with_previous": overlap}
checkedv5 = pd.DataFrame(data)



#Checking my results
def compare_results(original, my_version, column_name):
    similarity= []
    for i in range(0, length):
        if original[f"{column_name}"].iloc[i] == my_version[f"{column_name}"].iloc[i]:
            same = True
        else:
            same = "NOT SAME"
        similarity.append(same)
    return similarity


data1 = {'similarity':compare_results(df, checkedv5, "same_input_as_previous") }#, "overlapped similarity": compare_results(df, checkedv5, "overlapped_with_previous")}
comparison = pd.DataFrame(data1)

print(comparison.loc[comparison['similarity'] == 'NOT SAME'])
#print(comparison.loc[comparison['overlapped similarity'] == 'NOT SAME'])

#plot them
fig, ax = plt.subplots(figsize =(15,2))
for i in range(len(df)):
    if i%2 == 0:
        color = 'blue'
    else:
        color = 'red'
    ax.plot([df.input.iloc[i], df.output.iloc[i]], [0.0, 0.0], marker = '', color = color)

ax.scatter(df.input, np.zeros_like(df.input), color= 'lime')
ax.scatter(df.output, np.zeros_like(df.input), color = 'cyan')

plt.show()