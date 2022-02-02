import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# 2D continuity with overlaps

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_2d_v4.pkl')
dff = df.copy()
print(df.head)
print(df.columns)

def check_2d_continuity(segment1, segment2, verbose= False):
    distance_vector = segment2 - segment1
    cutoff = 1e-4
    distance_x = distance_vector[0]
    distance_y = distance_vector[1]
    distance_total = np.linalg.norm(distance_vector)
    #if abs(segment2[0][0] - segment1[1][1]) <= cutoff and abs(segment2[0][1] - segment1[1][1]) <= cutoff:
    if distance_total <= cutoff:
        continuity = True
    else:
        continuity = False
    if verbose:
        print(segment1, segment2)
        print("Distance X:", distance_x)
        print("Distance Y:",distance_y)
    return continuity

def check_2d_overlap(vector1, vector2):
    dot = np.dot(vector1, vector2)
    if dot < 0:
        overlap = True
    else:
        overlap = False
    return overlap

    '''
    distance_vector =  segment1 - segment2
    distance_x = distance_vector[0]
    distance_y = distance_vector[1]
    distance_total = np.linalg.norm(distance_vector)
    if distance_x or distance_y<0:
        overlap = True
    else:
        overlap = False
    return overlap
    '''

length = len(df)
overlap = []
continuity = []

for i in range(0,length):
    if i ==  0:
        x = True
        y = False
        continuity.append(x)
        overlap.append(y)
    else:
        segment1_input = np.array([df['input_x'].iloc[i-1], df['input_y'].iloc[i-1]])
        segment1_output = np.array([df['output_x'].iloc[i-1], df['output_y'].iloc[i-1]])
        segment2_input = np.array([df['input_x'].iloc[i], df['input_y'].iloc[i]])
        segment2_output = np.array([df['output_x'].iloc[i], df['output_y'].iloc[i]])
        vector1 = [df['output_x'].iloc[i - 1] - df['input_x'].iloc[i - 1], df['output_y'].iloc[i - 1] - df['input_y'].iloc[i - 1]]
        vector1 = np.array(vector1)
        distancevector = [df['input_x'].iloc[i] - df['output_x'].iloc[i-1],df['input_y'].iloc[i] - df['output_y'].iloc[i-1]]
        distancevector = np.array(distancevector)
        x = check_2d_continuity(segment1_output, segment2_input, verbose=False)
        y = check_2d_overlap(vector1, distancevector)
        continuity.append(x)
        overlap.append(y)


data = {'connected_to_previous': continuity, 'overlapped_with_previous': overlap}
testv4df = pd.DataFrame(data)
print(testv4df)

print(df['connected_to_previous'] == testv4df['connected_to_previous'])
print(df['overlapped_with_previous'] == testv4df['overlapped_with_previous'])
print((df['overlapped_with_previous'] == testv4df['overlapped_with_previous']).sum(), len(df))
#print(testv4df['overlapped_with_previous'].iloc[:50])
#print(df['overlapped_with_previous'].iloc[:50])
print(df[df['overlapped_with_previous'] != testv4df['overlapped_with_previous']])

#Plotting


fig, ax = plt.subplots(figsize =(15,2))
for i in range(len(df)):
    if i%2 == 0:
        color = 'blue'
    else:
        color = 'red'
        
    ax.plot([df['input_x'].iloc[i], df['output_x'].iloc[i]], [df['input_y'].iloc[i], df['output_y'].iloc[i]], marker = '', color = color)

ax.scatter(df['input_x'], df['input_y'], color = 'lime')  
ax.scatter(df['output_x'], df['output_y'], color = 'cyan')

plt.show()