import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from Plot_update import *

geom_df_mu2e = pd.read_pickle('/home/shared_data/helicalc_params/Mu2e_Coils_Conductors.pkl')
x, y, z = get_thick_cylinder_surface_xyz(geom_df_mu2e, 1)

print(geom_df_mu2e[:])

cyl = go.Figure(data = go.Surface(x=x, y=y, z=z,
                 showscale =False,
                 showlegend = False,
                 name = 'Coils (radial center)'
                 ))


cyl.show()

