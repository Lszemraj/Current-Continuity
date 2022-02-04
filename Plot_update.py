import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def cylinder(r, h, xc=0, yc=0, zc=0, pitch=0., yaw=0., roll=0., nt=100, nv=50, flip_angles=False):
    # generate grid of theta, z (or v for vertical)
    theta = np.linspace(2*np.pi+np.pi/4, 0+np.pi/4, nt)
    v = np.linspace(-h/2, h/2, nv)
    # make 2D grid for plotting later on
    TH, VV = np.meshgrid(theta, v)
    # cylindrical --> cartesian
    x = (r*np.cos(TH)).flatten()
    y = (r*np.sin(TH)).flatten()
    z = (VV).flatten()
    pos = np.array([x,y,z])
    # apply any coil rotations, w.r.t. cylinder center
    rot_angles = np.array([pitch, yaw, roll])
    # set to true or false depending on rotation sign definitions
    if flip_angles:
        rot_angles *= -1
    rot = Rotation.from_euler('XYZ', rot_angles, degrees=True)
    pos_rot = rot.apply(pos.T).T
    # get rotated x, y, z and translate to correct location
    x = pos_rot[0] + xc
    y = pos_rot[1] + yc
    z = pos_rot[2] + zc
    # reshape to 2D arrays for surface plotting
    x = x.reshape((len(v), len(theta)))
    y = y.reshape((len(v), len(theta)))
    z = z.reshape((len(v), len(theta)))
    return x, y, z

def get_cylinder_inner_surface_xyz(df, coil_num):
    c = df.query(f'Coil_Num == {coil_num}').iloc[0]
    x, y, z = cylinder(c.Ro, c.L, xc=c.x, yc=c.y, zc=c.z,
                       pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                       nt=360, nv=3, flip_angles=False)
    return x,y,z

def get_cylinder_outer_surface_xyz(df, coil_num):
    c = df.query(f'Coil_Num == {coil_num}').iloc[0]
    x, y, z = cylinder(c.Ro, c.L, xc=c.x, yc=c.y, zc=c.z,
                       pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                       nt=360, nv=3, flip_angles=False)
    return x,y,z

def get_thick_cylinder_surface_xyz(df, coil_num):
    c = df.query(f'Coil_Num == {coil_num}').iloc[0]
    # inner shell
    xi, yi, zi = cylinder(c.Ri, c.L, xc=c.x, yc=c.y, zc=c.z,
                          pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                          nt=360, nv=3, flip_angles=False)
    # outer shell
    xo, yo, zo = cylinder(c.Ro, c.L, xc=c.x, yc=c.y, zc=c.z,
                          pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                          nt=360, nv=3, flip_angles=False)
    # combine inner and outer layers
    x = np.concatenate([xo, xi[::-1], xo[:1]], axis=0)
    y = np.concatenate([yo, yi[::-1], yo[:1]], axis=0)
    z = np.concatenate([zo, zi[::-1], zo[:1]], axis=0)
    return x,y,z

datadir = '/home/shared_data/helicalc_params/'

def load_data(filename):
    df_raw = pd.read_pickle(datadir + f"{filename}")
    return df_raw
