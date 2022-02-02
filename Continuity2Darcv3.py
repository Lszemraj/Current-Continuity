import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

initial_x = []
initial_y = []
output_y = []
output_x = []
continuity = []
flip = []

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


def check_2d_flip(vector1, vector2, segment1_output, segment2_input, verbose):
        x = check_2d_continuity(segment1_output, segment2_input, verbose = False)
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

def check_arcsegment_continuity_flip(df, i):
    segment1_input = np.array([df['x0'].iloc[i - 1], df['y0'].iloc[i - 1]])
    segment1_output = np.array([df['xf'].iloc[i - 1], df['yf'].iloc[i - 1]])
    segment2_input = np.array([df['x0'].iloc[i], df['y0'].iloc[i]])
    segment2_output = np.array([df['xf'].iloc[i], df['yf'].iloc[i]])
    vector1 = [df['xf'].iloc[i - 1] - df['x0'].iloc[i - 1], df['yf'].iloc[i - 1] - df['y0'].iloc[i - 1]]
    vector1 = np.array(vector1)
    vector2 = [df['xf'].iloc[i] - df['x0'].iloc[i] , df['yf'].iloc[i] - df['y0'].iloc[i]]
    vector2 = np.array(vector2)
    #x = check_2d_continuity(segment1_output, segment2_input, verbose = False)
    connected = []
    for i in [segment1_input,segment1_output]:
        for n in [segment2_input,segment2_output]:
            x = check_2d_continuity(i, n, verbose=False)
            connected.append(x)
        if np.any(connected):
            x = True
        else:
            x = False
    y = check_2d_flip(vector1, vector2, segment1_output, segment2_input, verbose=False)
    continuity.append(x)
    flip.append(y)


print("Continuity2Darcv1 __name__ is set to: {}".format(__name__))

if __name__ == "__main__":
    df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/arc_segments_2d_v3.pkl')
    dff = df.copy()
    #print(df.columns)

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
            flip.append(y)
        else:
            check_arcsegment_continuity_flip(arcv2df, i)

    # Make dataframe

    data2 = {'x0': initial_x, 'y0': initial_y, 'xf': output_x, 'yf': output_y, 'connected_to_previous': continuity, 'flipped_with_previous': flip}
    arcdf = pd.DataFrame(data2)

    # Check and compare values
    print("continuity check:", (df['connected_to_previous'] == arcdf['connected_to_previous']).sum(), len(df))
    print(df[df['connected_to_previous'] != arcdf['connected_to_previous']])
    print("flipped check:", (df['flipped_to_previous'] == arcdf['flipped_with_previous']).sum(), len(df))
    print(df[df['flipped_to_previous'] != arcdf['flipped_with_previous']])