import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


#testing 2D cases with discontinuities and flipped vectors
df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_2d_v3.pkl')
dff = df.copy()
print(df.head)
print(df.columns)


def check_2d_flip(vector1, vector2, segment1_output, segment2_input, verbose):
        x = check_2d_continuity(segment1_output, segment2_input, verbose = True)
        if x :
            print(vector1, vector2, "Connected, Not Flipped")
            return False
        dot = np.dot(vector1, vector2)
        if verbose:
            print(f"{vector1} dot {vector2} equals {dot}")
        if dot >= 0:
            if verbose:
                print('Not Flipped')
            flipped = False
        else:  # if np.dot(vector1, vector2) < 0:
            flipped = True
        # else:
        # raise TypeError("Not 1D segments")
        if verbose:
            print(flipped)
        return flipped

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

length = len(df)
continuity = []
flipped = []

for i in range(0,length):
    if i ==  0:
        x = True
        y = False
        continuity.append(x)
        flipped.append(y)
    else:
        print(i)
        segment1_input = np.array([df['input_x'].iloc[i-1], df['input_y'].iloc[i-1]])
        segment1_output = np.array([df['output_x'].iloc[i-1], df['output_y'].iloc[i-1]])
        segment2_input = np.array([df['input_x'].iloc[i], df['input_y'].iloc[i]])
        segment2_output = np.array([df['output_x'].iloc[i], df['output_y'].iloc[i]])
        vector1 = df['output_x'].iloc[i - 1] - df['input_x'].iloc[i - 1]
        vector1 = np.array([vector1])
        vector2 = df['output_x'].iloc[i] - df['input_x'].iloc[i]
        vector2 = np.array([vector2])
        x = check_2d_continuity(segment1_output, segment2_input, verbose=False)
        y = check_2d_flip(vector1, vector2,segment1_output, segment2_input, verbose = False)
        continuity.append(x)
        flipped.append(y)

data = {'connected_to_previous': continuity, 'flipped_with_previous': flipped}
testv3df = pd.DataFrame(data)
print(testv3df)

print(df['connected_to_previous'] == testv3df['connected_to_previous'])
print(df['flipped_to_previous'] == testv3df['flipped_with_previous'])

print((df['flipped_to_previous'] == testv3df['flipped_with_previous']).sum(), len(df))

print(df[df['flipped_to_previous'] != testv3df['flipped_with_previous']])