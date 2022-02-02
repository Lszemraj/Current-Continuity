import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#Chain of 3D segments, shuffled order

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_3d_v7.pkl')

dff = df.copy()
print(df.head)
print(df.columns)

