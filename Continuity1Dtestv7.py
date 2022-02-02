import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v7.pkl')

print(df.head)

def check_1d_continuity(segment1, segment2):
    cutoff = 1e-4
    if abs(segment2[0] - segment1[1]) <= cutoff:
        continuity = True
    else:
        continuity = False
    return continuity

def put_in_correct_order(segment1, segment2):
    cutoff = 3
    if segment2[0] - segment1[1] <= cutoff:
        next_in_line = True
    else:
        next_in_line = False
    return next_in_line

#New dataframe with just input, output
data = [df["input"], df["output"]]
headers = ["input", "output"]
df2 = pd.concat(data, axis=1, keys=headers)
print("df2:", df2.head)

sorted = df2.sort_values("input"
    #by=["input", "output"],
    #ascending = [True, True]
)


def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]

x = find_neighbours(df["output"].iloc[0], df, "input")
print(x)
print(df.iloc[171])
dff = df.copy()
#list = np.zeros(len(df))
dff["Neighbours"] = 0

for i in range(0, len(df)):
    x = find_neighbours(df["output"].iloc[i], dff, "input")
    dff["Neighbours"].iloc[i] = x

print(dff)









'''
close_list = []
input = []
output = []

for i in range(0, len(df)):
    #print(type(segment))
    seg1 = (df2["input"].iloc[i], df2["output"].iloc[i])
    df2["Next in Line"] = np.where(abs(df2["input"] - df["output"].iloc[i] ) < 3, 'True', 'False')
    when df2.query("Next in Line" == True):
        x = (df2["Next in Line"] == True).index
        input.append(df2["input"].iloc[x])
        output.append(df2["input"].iloc[x])
'''





'''
    for z in range(0, len(df)):
        close_or_not = put_in_correct_order(x, z)
        if close_or_not:
            close = np.array([z, close_or_not])
            close_list.append(close)

            
'''

