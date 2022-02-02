import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#Chain of 3D segments, multiple connections

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_3d_v5.pkl')

dff = df.copy()
print(df.head)
print(df.columns)



def check_same_input_3D(segment1, segment2):
    cutoff = 1e-4
    if abs(segment1[0] - segment2[0])<= cutoff and  abs(segment1[1] - segment2[1])<= cutoff and abs(segment1[2] - segment2[2])<= cutoff:
        same_input = True
    else:
        same_input = False
    return same_input


def check_3d_continuity(segment1, segment2, verbose= False):
    distance_vector = segment2 - segment1
    cutoff = 1e-4
    distance_x = distance_vector[0]
    distance_y = distance_vector[1]
    distance_z = distance_vector[2]
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





length = len(df)
same_input_as_previous = []
continuity = []

for i in range(0,length):
    if i ==  0:
        x = True
        y = False
        continuity.append(x)
        same_input_as_previous.append(y)

    else:
        segment1_input = np.array([df['input_x'].iloc[i-1], df['input_y'].iloc[i-1],df['input_z'].iloc[i-1] ])
        segment1_output = np.array([df['output_x'].iloc[i-1], df['output_y'].iloc[i-1], df['output_z'].iloc[i-1]])
        segment2_input = np.array([df['input_x'].iloc[i], df['input_y'].iloc[i], df['input_z'].iloc[i]])
        segment2_output = np.array([df['output_x'].iloc[i], df['output_y'].iloc[i], df['output_z'].iloc[i]])
        #vector1 = df['output_x'].iloc[i - 1] - df['input_x'].iloc[i - 1]
        #vector1 = np.array([vector1])
        #vector2 = df['output_x'].iloc[i] - df['input_x'].iloc[i]
        #vector2 = np.array([vector2])
        x = check_3d_continuity(segment1_output, segment2_input, verbose=False)
        y = check_same_input_3D(segment1_input, segment2_input)

        continuity.append(x)
        same_input_as_previous.append(y)



data = {'connected_to_previous': continuity, "same_input_as_previous": same_input_as_previous}
testv5df = pd.DataFrame(data)

#print(df['connected_to_previous'] ==  testv2df['connected_to_previous'])
#print((df['connected_to_previous'] ==  testv5df['connected_to_previous']).sum(), len(df))
print((df['same_input_as_previous'] ==  testv5df['same_input_as_previous']).sum(), len(df))
print(df[df['same_input_as_previous'] != testv5df['same_input_as_previous']])