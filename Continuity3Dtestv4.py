import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#Chain of 3D segments, possibly overlapped

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_3d_v4.pkl')

dff = df.copy()
print(df.head)
print(df.columns)

def check_3d_continuity(segment1, segment2, verbose= False):
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

def check_3d_overlap(segment1_output, segment2_input, vector1, vector2, verbose = False):
    dot = np.dot(vector1, vector2)
    x = check_3d_continuity(segment1_output, segment2_input)
    if verbose:
        print(f"{vector1} dot {vector2} equals {dot}")
    len1 = np.linalg.norm(vector1)
    len2 = np.linalg.norm(vector2)
    dot_180 = len1*len2*(-1)
    if x:
        overlap = False
    elif np.isclose(dot, dot_180):
        overlap = True
    else:
        overlap = False
    return overlap

#Calculate
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
        print(i)
        segment1_input = np.array([df['input_x'].iloc[i-1], df['input_y'].iloc[i-1], df['input_z'].iloc[i-1]])
        segment1_output = np.array([df['output_x'].iloc[i-1], df['output_y'].iloc[i-1], df['output_z'].iloc[i-1]])
        segment2_input = np.array([df['input_x'].iloc[i], df['input_y'].iloc[i], df['input_z'].iloc[i]])
        segment2_output = np.array([df['output_x'].iloc[i], df['output_y'].iloc[i],  df['output_z'].iloc[i]])
        vector1 = [df['output_x'].iloc[i - 1] - df['input_x'].iloc[i - 1], df['output_y'].iloc[i - 1] - df['input_y'].iloc[i - 1], df['output_z'].iloc[i-1] - df['input_z'].iloc[ i -1]]
        vector1 = np.array(vector1)
        distancevector = [df['input_x'].iloc[i] - df['output_x'].iloc[i-1],df['input_y'].iloc[i] - df['output_y'].iloc[i-1], df['input_z'].iloc[i] - df['output_z'].iloc[i-1]]
        distancevector = np.array(distancevector)
        x = check_3d_continuity(segment1_output, segment2_input, verbose=False)
        y = check_3d_overlap(segment1_output, segment2_input, vector1, distancevector, verbose = True)
        continuity.append(x)
        overlap.append(y)


data = {'connected_to_previous': continuity, 'overlapped_with_previous': overlap}
testv4df = pd.DataFrame(data)
print(testv4df)

print((df['overlapped_with_previous'] == testv4df['overlapped_with_previous']).sum(), len(df))
print(df[df['overlapped_with_previous'] != testv4df['overlapped_with_previous']])

#print(df[:20])
#Throwing errors in all cases except 161 because neighbours are connected to previous but the point itself is not