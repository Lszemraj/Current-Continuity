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

geom_df_mu2e = load_data("Mu2e_Coils_Conductors.pkl")

def get_many_thick_cylinders(df):
    # lists to store all cylinder information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all coils, by coil number
    for cn in df.Coil_Num:
        # get x,y,z for this cylinder
        x_, y_, z_ = get_thick_cylinder_surface_xyz(geom_df_mu2e, cn)  # inner and outer walls
        # pad x_, y_, z_ for transparency between coils
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        # color array, with padded x,y,z flipped to zero (will set transparent)
        cs_ = np.ones_like(x_)
        cs_[0, :] = 0
        cs_[-1, :] = 0
        # add to lists
        xs.append(x_)
        ys.append(y_)
        zs.append(z_)
        cs.append(cs_)
    # create numpy arrays from gathered results
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    cs = np.concatenate(cs)

    return xs, ys, zs, cs

def create_bar_endpoints(df_bars, index):
    x0 = df_bars['x0'].iloc[index]
    y0 = df_bars['y0'].iloc[index]
    z0 = df_bars['z0'].iloc[index]
    zf = x0 = df_bars['z0'].iloc[index] + df_bars['length'].iloc[index]
    width = df_bars['W'].iloc[index]
    Thickness = df_bars['T'].iloc[index]
    T2 = Thickness / 2
    W2 = width / 2
    xc = [x0 + W2, x0 + W2, x0 - W2, x0 - W2, x0 + W2, x0 + W2, x0 - W2, x0 - W2]
    yc = [y0 + T2, y0 - T2, y0 - T2, y0 + T2, y0 + T2, y0 - T2, y0 - T2, y0 + T2]
    zc = [z0, z0, z0, z0, zf, zf, zf, zf]
    return xc, yc, zc