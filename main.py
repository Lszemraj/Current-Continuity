import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#File
df = pd.read_pickle("/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl")

#Plotly attempt??
def cylinder(r, L, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(-L/2, L/2, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z


from mpl_toolkits.mplot3d import Axes3D


# Radius and length values
r = 2
L = 10



# Plot that cylinder
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y, Z =  cylinder(r, L)
#ax.plot_surface(X, Y, Z)
#plt.show()

#translations to get to Xc, Yc, Zc
#This moves the very center of the cylinder!

Xc = 10
Yc = 0
Zc = 20

X = X + Xc
Y = Y + Yc
Z = Z + Zc

ax.plot_surface(X, Y, Z)
#plt.show()


#Rotations

rot_angle = np.array([0,3,0])
rot = Rotation.from_euler('XYZ', rot_angles, degrees= True)
pos_rot = rot.apply(pos.T)