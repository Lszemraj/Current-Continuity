import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#Collection of random arc segments in 2D

#need to get input and output points

initial_x = []
initial_y = []
output_y = []
output_x = []


def find_2D_arcpoints(Cx, Cy, r, angle, delta_angle):
    X = Cx + (r * np.cos(angle))
    Y = Cy + (r * np.sin(angle))
    X_final = Cx + (r * np.cos(angle + delta_angle))
    Y_final = Cy + (r * np.sin(angle + delta_angle))
    return X, Y, X_final, Y_final

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

print("Continuity2Darcv1 __name__ is set to: {}" .format(__name__))

if __name__ == "__main__":
    df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/arc_segments_2d_v1.pkl')
    dff = df.copy()

    length = len(dff)

    for i in range(0, length):
        check_arcsegments(dff, i)

    #Make dataframe
    data = {'x0': initial_x, 'y0': initial_y, 'xf': output_x, 'yf': output_y}
    arcv1df = pd.DataFrame(data)

    #Check and compare values
    print((df['x0_true'] == arcv1df['x0']).sum(), len(df))
    print(df[df['x0_true'] != arcv1df['x0']])