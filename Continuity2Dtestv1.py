import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_2d_v2.pkl')
dff = df.copy()
print(df.head)

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

# [[inputx, inputy], [outputx, outputy]]
continuity = []
length = len(df)



for i in range(0,length):
    if i ==  0:
        x = True
        continuity.append(x)
    else:
        segment1_input = np.array([df['input_x'].iloc[i-1], df['input_y'].iloc[i-1]])
        segment1_output = np.array([df['output_x'].iloc[i-1], df['output_y'].iloc[i-1]])
        segment2_input = np.array([df['input_x'].iloc[i], df['input_y'].iloc[i]])
        segment2_output = np.array([df['output_x'].iloc[i], df['output_y'].iloc[i]])
        x = check_2d_continuity(segment1_output, segment2_input, verbose=True)
        continuity.append(x)

dff = pd.DataFrame(continuity)
dff.columns = ["continuity"]

print(df["connected_to_previous"] ==  dff["continuity"])
#print(df["connected_to_previous"])
#print(dff["continuity"])