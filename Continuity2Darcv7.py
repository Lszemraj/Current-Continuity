import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Collection of random arc segments in 2D
# need to get input and output points


#Ordering segments
initial_x = []
initial_y = []
output_y = []
output_x = []
continuity = []

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

def check_arcsegment_continuity(df, i):
    segment1_input = np.array([df['x0'].iloc[i - 1], df['y0'].iloc[i - 1]])
    segment1_output = np.array([df['xf'].iloc[i - 1], df['yf'].iloc[i - 1]])
    segment2_input = np.array([df['x0'].iloc[i], df['y0'].iloc[i]])
    segment2_output = np.array([df['xf'].iloc[i], df['yf'].iloc[i]])
    x = check_2d_continuity(segment1_output, segment2_input, verbose = False)
    continuity.append(x)



def check_order(df, length):
    index_list = []
    dff = df.copy()
    while len(dff) > 0:
        i = 0
        pos_out = np.array([df["xf"].iloc[i], df['yf'].iloc[i]])
        pos_in = np.array([df["x0"].iloc[i], df['y0'].iloc[i]])
        distances = ((df['x0'] - pos_out[0])**2 + (df['y0'] - pos_out[1])**2)**(1/2)
        closest = np.argmin(distances) #index of closest
        #if closest > 20:
           # distances2 = ((df['xf'] - pos_in[0])**2 + (df['yf'] - pos_in[1])**2)**(1/2)
           # closest2 = np.argmin(distances2)
          #  index_list.insert(0, closest2)
        #else:
        index_list.append(closest)
        i = i + closest
        dff.drop(closest)
    data = {"sorted_order": index_list}
    dff = pd.DataFrame(data)
    return dff



print("Continuity2Darcv1 __name__ is set to: {}".format(__name__))

if __name__ == "__main__":
    df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/arc_segments_2d_v7.pkl')
    dff = df.copy()

    length = len(dff)

    for i in range(0, length):
        check_arcsegments(dff, i)
    correct_order  = df["correct_order"][:]

    data = {'x0': initial_x, 'y0': initial_y, 'xf': output_x, 'yf': output_y, "correct_order": correct_order }
    arcv2df = pd.DataFrame(data)

    for i in range(0,length):
        if i == 0:
            x = True
            continuity.append(x)
        else:
            check_arcsegment_continuity(arcv2df, i)
    #print(arcv2df)
    #print(len(arcv2df))
    arc = check_order(arcv2df, length)
    print("sorted", arc)
    print(df["correct_order"])
    # Make dataframe

    data2 = {'x0': initial_x, 'y0': initial_y, 'xf': output_x, 'yf': output_y, 'connected_to_previous': continuity}
    arcdf = pd.DataFrame(data2)

    # Check and compare values
    print((df['correct_order'] == arc['sorted_order']).sum(), len(df))
    #print(df[df['connected_to_previous'] != arcdf['connected_to_previous']])
    #print(df)