import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v6.pkl')
print(df.iloc[0])
print(df.columns)

# plot
fig, ax = plt.subplots(figsize =(15,2))
for i in range(len(df)):
    if i%2 == 0:
        color = 'blue'
    else:
        color = 'red'
    ax.plot([df.input.iloc[i], df.output.iloc[i]], [i*0.1, i*0.1], marker = '', color = color)

ax.scatter(df.input, 0.1*np.arange(len(df)), color= 'lime')
ax.scatter(df.output, 0.1*np.arange(len(df)), color = 'cyan')



# function

def check_all(segment1, segment2):
    #check if same input
    cutoff = 1e-4
    if abs(segment1[0] - segment2[0]) < cutoff:
        same_input = True
        #continuity = True
    else:
        same_input = False
        #if abs(segment2[0] - segment1[1]) <= cutoff:
            #continuity = True
        #else:
            #continuity = False

    #check if flipped
    vector1 = np.array([segment1[1]-segment1[0]])
    vector2 = np.array([segment2[1]-segment2[0]])
    if np.dot(vector1, vector2) >= 0:
        flipped = False
    else: #if np.dot(vector1, vector2) < 0:
        flipped = True
        
    if abs(segment2[0] - segment1[1])<= cutoff:
        continuity = True
    else:
        continuity = False
    if segment2[0] - segment1[1] < 0:
        overlap = True
    else: #if segment2[0] - segment1[1] >= 0:
        overlap = False
    return continuity, flipped, overlap, same_input 


    #check if overlapped or connected
length = len(df)

continuity = []
overlap = []
flipped = []
input = []

for i in range(0,length):
    print(i)
    if i ==  0:
        x = True
        y = False
        z = False
        f = False
        continuity.append(x)
        flipped.append(y)
        overlap.append(z)
        input.append(f)
    else:
        segment1 = [df['input'].iloc[i-1], df['output'].iloc[i-1]]
        segment2 = [df['input'].iloc[i], df['output'].iloc[i]]
        x,y, z, f = check_all(segment1, segment2)
        continuity.append(x)
        flipped.append(y)
        overlap.append(z)
        input.append(f)
print("out of loop")
print(continuity)
print(overlap)
print(flipped)
print(input)

data = {'continuity': continuity, 'overlapped': overlap, "flipped": flipped, "same_input": input}  
dfv6 = pd.DataFrame(data)
print(dfv6)

print(df['connected_to_previous'] == dfv6['continuity'])
print((df['connected_to_previous'] == dfv6['continuity']).sum(), len(df))

#print(df['same_input_as_previous'] == dfv6['same_input'])
#print((df['same_input_as_previous'] == dfv6['same_input']).sum(), len(df))

#print(df['overlapped_with_previous'] == dfv6['overlapped'])
#print((df['overlapped_with_previous'] == dfv6['overlapped']).sum(), len(df))


plt.show()
