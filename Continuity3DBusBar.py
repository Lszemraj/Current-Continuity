import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from Plot_update import *
import dash
from dash import Dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html

#Check Continuity of 3D Rectangular bus bar cross section

df_dict = load_all_geoms(return_dict=True)
df_str = df_dict['straights']
#print("df_str", df_str.columns)
df_arcs = df_dict['arcs']
#print("df_arcs", df_arcs.columns)


def create_bar_endpoints_no_rot(df_bars, index):
    x0 = df_bars['x0'].iloc[index]
    y0 = df_bars['y0'].iloc[index]
    z0 = df_bars['z0'].iloc[index]
    theta2 = df_bars["theta2"].iloc[index]
    length = df_bars['length'].iloc[index]
    if theta2 < 1:
        scale = 1
    else:
        scale = -1
    zf = df_bars['z0'].iloc[index] + (df_bars['length'].iloc[index])*scale
    width = df_bars['W'].iloc[index]
    Thickness = df_bars['T'].iloc[index]
    T2 = Thickness / 2
    W2 = width / 2
    xc = [W2, W2, -W2, -W2, W2, W2, -W2, -W2]
    yc = [T2, -T2, -T2, T2, T2, -T2, -T2, T2]
    zc = [-length / 2, -length / 2, -length / 2, -length / 2, length / 2, length / 2, length / 2, length / 2]

    pos = np.array([xc, yc, zc])
    phi2 = df_bars["Phi2"].iloc[index]
    theta2 = df_bars["theta2"].iloc[index]
    psi2 = df_bars["psi2"].iloc[index]

    X = xc + x0
    Y = yc + y0
    Z = zc + z0

    return X, Y, Z



def check_if_planes_parallel(df, index):
    X1, Y1, Z1 = create_bar_endpoints_no_rot(df, index -1)
    face1 = np.array([[X1[0], Y1[0], Y1[0]], [X1[1], Y1[1], Y1[1]], [X1[2], Y1[2], Y1[2]], [X1[3], Y1[3], Y1[3]]])
    #
    X2, Y2, Z2 = create_bar_endpoints_no_rot(df, index)
    face2 = np.array([[X2[0], Y2[0], Y2[0]], [X2[1], Y2[1], Y2[1]], [X2[2], Y2[2], Y2[2]], [X2[3], Y2[3], Y2[3]]])


    # face 1
    v1_1 = face1[1] - face1[0]
    v2_1 = face1[2] - face1[0]
    # face 2
    v1_2 = face2[1] - face2[0]
    v2_2 = face2[2] - face2[0]
    # face 1
    normal_vec_1 = np.cross(v2_1, v1_1)
    normal_vec_1 = normal_vec_1 / np.linalg.norm(normal_vec_1)
    # face 2
    normal_vec_2 = np.cross(v2_2, v1_2)
    normal_vec_2 = normal_vec_2 / np.linalg.norm(normal_vec_2)

    if np.allclose(normal_vec_1, normal_vec_2):
        parallel = True
    else:
        parallel = False
    return parallel

def check_distance_normal(df, index):
    X1, Y1, Z1 = create_bar_endpoints_no_rot(df, index - 1)
    face1 = np.array([[X1[0], Y1[0], Y1[0]], [X1[1], Y1[1], Y1[1]], [X1[2], Y1[2], Y1[2]], [X1[3], Y1[3], Y1[3]]])
    #
    X2, Y2, Z2 = create_bar_endpoints_no_rot(df, index)
    face2 = np.array([[X2[0], Y2[0], Y2[0]], [X2[1], Y2[1], Y2[1]], [X2[2], Y2[2], Y2[2]], [X2[3], Y2[3], Y2[3]]])

    # face 1
    v1_1 = face1[1] - face1[0]
    v2_1 = face1[2] - face1[0]
    # face 2
    v1_2 = face2[1] - face2[0]
    v2_2 = face2[2] - face2[0]

    # face 1
    normal_vec_1 = np.cross(v2_1, v1_1)
    normal_vec_1 = normal_vec_1 / np.linalg.norm(normal_vec_1)
    # face 2
    normal_vec_2 = np.cross(v2_2, v1_2)
    normal_vec_2 = normal_vec_2 / np.linalg.norm(normal_vec_2)

    center_1 = face1[0] + v1_1 / 2 + v2_1 / 2
    center_2 = face2[0] + v1_2 / 2 + v2_2 / 2
    difference_vec = center_2 - center_1
    normal_dist = np.dot(difference_vec, normal_vec_1)
    parallel_dist = (np.linalg.norm(difference_vec) ** 2 - np.linalg.norm(normal_dist) ** 2) ** (1 / 2)
    cutoff_parallel = 1 / 2*df['W'].iloc[index]

    if normal_dist > 1e-4:
        connected = False
    else:
        if parallel_dist < cutoff_parallel:
            connected = True
        else:
            connected = False
    return connected


#Checking if parallel
parallel_list = []
for i in range(0, len(df_str)):
    if i == 0:
        pass
    else:
        parallel = check_if_planes_parallel(df_str, i)
        parallel_list.append(parallel)

#print(parallel_list)


#Checking if connected
connected_list = []

for i in range(0, len(df_str)):
    if i == 0:
        pass
    else:
        connected = check_distance_normal(df_str, i)
        connected_list.append(connected)

print("connected_list", connected_list)