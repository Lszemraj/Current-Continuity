import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Collection of random arc segments in 2D
# need to get input and output points


# Checking multiple connections

initial_x = []
initial_y = []
output_y = []
output_x = []
continuity = []
multiple_connections = []

def find_2D_arcpoints(Cx, Cy, r, angle, delta_angle):
    X = Cx + (r * np.cos(angle))
    Y = Cy + (r * np.sin(angle))
    X_final = Cx + (r * np.cos(angle + delta_angle))
    Y_final = Cy + (r * np.sin(angle + delta_angle))
    return X, Y, X_final, Y_final


def check_2d_continuity(segment1, segment2, verbose= False):
    distance_vector = segment2 - segment1
    cutoff = 1e-4
    distance_x = distance_vector[0]
    distance_y = distance_vector[1]
    distance_total = np.linalg.norm(distance_vector)
    if distance_total <= cutoff:
        continuity = True
    else:
        continuity = False
    if verbose:
        print(segment1, segment2)
        print("Distance X:", distance_x)
        print("Distance Y:",distance_y)
    return continuity

def check_same_input_2D(segment1, segment2):
    cutoff = 1e-4
    if abs(segment1[0] - segment2[0])<= cutoff and  abs(segment1[1] - segment2[1])<= cutoff:
        same_input = True
    else:
        same_input = False
    return same_input

def check_arcsegments(df, i):
    Cx = df['xc'].iloc[i]
    Cy = df['yc'].iloc[i]
    r = df['R'].iloc[i]
    angle = df['phi0'].iloc[i]
    delta_angle = df['dphi'].iloc[i]
    x0, y0, xf, yf = find_2D_arcpoints(Cx, Cy, r, angle, delta_angle)
    initial_x.append(x0)
    initial_y.append(y0)
    output_x.append(xf)
    output_y.append(yf)

def check_arcsegment_continuity_connections(df, i):
    segment1_input = np.array([df['x0'].iloc[i - 1], df['y0'].iloc[i - 1]])
    segment1_output = np.array([df['xf'].iloc[i - 1], df['yf'].iloc[i - 1]])
    segment2_input = np.array([df['x0'].iloc[i], df['y0'].iloc[i]])
    segment2_output = np.array([df['xf'].iloc[i], df['yf'].iloc[i]])
    x = check_2d_continuity(segment1_output, segment2_input, verbose = False)
    y = check_same_input_2D(segment1_input, segment2_input)
    continuity.append(x)
    multiple_connections.append(y)


print("Continuity2Darcv1 __name__ is set to: {}".format(__name__))

if __name__ == "__main__":
    df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/arc_segments_2d_v5.pkl')
    dff = df.copy()

    length = len(dff)

    for i in range(0, length):
        check_arcsegments(dff, i)

    data = {'x0': initial_x, 'y0': initial_y, 'xf': output_x, 'yf': output_y}
    arcv2df = pd.DataFrame(data)

    for i in range(0,length):
        if i == 0:
            x = True
            y = False
            continuity.append(x)
            multiple_connections.append(y)
        else:
            check_arcsegment_continuity_connections(arcv2df, i)

    # Make dataframe

    data2 = {'x0': initial_x, 'y0': initial_y, 'xf': output_x, 'yf': output_y, 'connected_to_previous': continuity,
             'same_input_as_previous': multiple_connections}
    arcdf = pd.DataFrame(data2)

    # Check and compare values
    print((df['same_input_as_previous'] == arcdf['same_input_as_previous']).sum(), len(df))
    print(df[df['same_input_as_previous'] != arcdf['same_input_as_previous']])
    #print(df)