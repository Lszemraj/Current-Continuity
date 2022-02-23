import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

def load_data(name):
    data_dir = '/home/shared_data/helicalc_params/'
    ddf = pd.read_pickle(data_dir +name)
    return ddf



# Loading dataframes with geometry information
def load_all_geoms(return_dict=True):
    # files
    geom_dir = '/home/shared_data/helicalc_params/'
    coils_file = geom_dir + 'Mu2e_Coils_Conductors.pkl'
    straight_file = geom_dir + 'Mu2e_Straight_Bars_V13.csv'
    arc_file = geom_dir + 'Mu2e_Arc_Bars_V13.csv'
    arc_transfer_file = geom_dir + 'Mu2e_Arc_Transfer_Bars_V13.csv'
    # load dataframes
    df_coils = pd.read_pickle(coils_file)
    df_str = pd.read_csv(straight_file)
    df_arc = pd.read_csv(arc_file)
    df_arc_tr = pd.read_csv(arc_transfer_file)
    if return_dict:
        df_dict = {'coils': df_coils, 'straights': df_str, 'arcs': df_arc,
                   'arcs_transfer': df_arc_tr}
        return df_dict
    else:
        return df_coils, df_str, df_arc, df_arc_tr

# Coils (idea cylinders)
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

def get_many_thick_cylinders(df):
    # lists to store all cylinder information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all coils, by coil number
    for cn in df.Coil_Num:
        # get x,y,z for this cylinder
        x_, y_, z_ = get_thick_cylinder_surface_xyz(df, cn)  # inner and outer walls
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

# bus bars
# straight sections (longitudinal and tangential)
def get_3d_straight(df, bar_num, nz=20):
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    euler2 = np.array([df_.Phi2, df_.theta2, df_.psi2])
    rot = Rotation.from_euler("zyz", angles=euler2[::-1], degrees=True) # extrinsic
    I_flow = df_.I_flow
    if np.isclose(I_flow, 0.):
        x0 = df_.x0
        y0 = df_.y0
        z0 = df_.z0
    else:
        x0 = df_.x1
        y0 = df_.y1
        z0 = df_.z0
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    # length
    L = df_.length
    # start in starting frame
    #zs = np.arange(z0, z0+L+1e-2, 1e-2)
    xs_list = []
    ys_list = []
    zs_list = []
    for dx, dy in zip([-W/2, -W/2, W/2, W/2], [-T/2, T/2, T/2, -T/2]):
        zs = np.linspace(0, L, nz)
        xs = dx*np.ones_like(zs)
        ys = dy*np.ones_like(zs)
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)
    xs_array = np.concatenate(np.array(xs_list).T)
    ys_array = np.concatenate(np.array(ys_list).T)
    zs_array = np.concatenate(np.array(zs_list).T)
    pos = np.array([xs_array, ys_array, zs_array]).T
    #return pos.T
    rpos = rot.apply(pos)
    #return rpos.T
    rtpos = rpos + np.array([[x0, y0, z0]])[-1,:]
    return rtpos.T

def get_3d_straight_surface(df, bar_num, nphi=20):
    xs, ys, zs = get_3d_straight(df, bar_num, nphi)
    N = nphi
    # do appropriate reshaping
    x_tot = []
    y_tot = []
    z_tot = []
    c_tot = []
    for ind0, ind1, o in zip([0,1,2,3], [1,2,3,0], [1, -1, 1, -1]):
#     for ind0, ind1, o in zip([0,1,2,3][::-1], [1,2,3,0][::-1], [1, -1, 1, -1]):
        x = xs.reshape((N, -1))[::o,[ind0,ind1]]
        y = ys.reshape((N, -1))[::o,[ind0,ind1]]
        z = zs.reshape((N, -1))[::o,[ind0,ind1]]
        c = np.ones_like(x)
        x_tot.append(x)
        y_tot.append(y)
        z_tot.append(z)
        c_tot.append(c)
    x_tot = np.concatenate(x_tot)
    y_tot = np.concatenate(y_tot)
    z_tot = np.concatenate(z_tot)
    c_tot = np.concatenate(c_tot)
    return x_tot, y_tot, z_tot, c_tot

def get_many_3d_straights(df):
    # lists to store all busbar information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all arcs, by conductor number
    for cn in df['cond N']:
        # get x,y,z for this cylinder
        x_, y_, z_, cs_ = get_3d_straight_surface(df, cn)
        # pad x_, y_, z_ for transparency between bars
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        cs_ = np.insert(np.insert(cs_, 0, cs_[0], axis=0), -1, cs_[-1], axis=0)
        # color array, with padded x,y,z flipped to zero (will set transparent)
        #cs_ = np.ones_like(x_)
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

# arcs
def get_3d_arc(df, bar_num, nphi=20):
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    euler2 = np.array([df_.Phi2, df_.theta2, df_.psi2])
    rot = Rotation.from_euler("zyz", angles=euler2[::-1], degrees=True) # extrinsic
    x0 = df_.x0
    y0 = df_.y0
    z0 = df_.z0
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    # R, dphi
    if 'R_curve' in df.columns:
        R = df_.R_curve # different setup for transfer line arcs
    else:
        R = df_.R0 # does not work only for last 4 arcs (to transfer line)
    PHI = df_.dphi
    # start in starting frame
    phis = np.linspace(0, np.radians(PHI), nphi)
    xs_list = []
    ys_list = []
    zs_list = []
    for dx, dy in zip([-W/2, -W/2, W/2, W/2], [-T/2, T/2, T/2, -T/2]):
        ys = -R * np.cos(phis) + R + dy
        zs = R * np.sin(phis)
        xs = np.zeros_like(phis) + dx
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)
    xs_array = np.concatenate(np.array(xs_list).T)
    ys_array = np.concatenate(np.array(ys_list).T)
    zs_array = np.concatenate(np.array(zs_list).T)
    pos = np.array([xs_array, ys_array, zs_array]).T
    #return pos.T
    rpos = rot.apply(pos)
    #return rpos.T
    rtpos = rpos + np.array([[x0, y0, z0]])[-1,:]
    return rtpos.T

def get_3d_arc_surface(df, bar_num, nphi=20):
    xs, ys, zs = get_3d_arc(df, bar_num, nphi)
    N = nphi
    # do appropriate reshaping
    x_tot = []
    y_tot = []
    z_tot = []
    c_tot = []
    for ind0, ind1, o in zip([0,1,2,3], [1,2,3,0], [1, -1, 1, -1]):
#     for ind0, ind1, o in zip([0,1,2,3][::-1], [1,2,3,0][::-1], [1, -1, 1, -1]):
        x = xs.reshape((N, -1))[::o,[ind0,ind1]]
        y = ys.reshape((N, -1))[::o,[ind0,ind1]]
        z = zs.reshape((N, -1))[::o,[ind0,ind1]]
        c = np.ones_like(x)
        x_tot.append(x)
        y_tot.append(y)
        z_tot.append(z)
        c_tot.append(c)
    x_tot = np.concatenate(x_tot)
    y_tot = np.concatenate(y_tot)
    z_tot = np.concatenate(z_tot)
    c_tot = np.concatenate(c_tot)
    return x_tot, y_tot, z_tot, c_tot

def get_many_3d_arcs(df):
    # lists to store all busbar information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all arcs, by conductor number
    for cn in df['cond N']:
        # get x,y,z for this cylinder
        x_, y_, z_, cs_ = get_3d_arc_surface(df, cn)
        # pad x_, y_, z_ for transparency between bars
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        cs_ = np.insert(np.insert(cs_, 0, cs_[0], axis=0), -1, cs_[-1], axis=0)
        # color array, with padded x,y,z flipped to zero (will set transparent)
        # cs_ = np.ones_like(x_)
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
    rot_angles = np.array([phi2, theta2, psi2])
    rot = Rotation.from_euler('ZYZ', rot_angles, degrees=True)
    pos_rot = rot.apply(pos.T)
    X, Y, Z = pos_rot.T
    X = X + x0
    Y = Y + y0
    Z = Z + z0

    return X, Y, Z