import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

df = pd.read_pickle('/home/shared_data/helicalc_params/conductivity_test/line_segments_1d_v3.pkl')
print(df.head)

#vectors
x_vec = np.array([1])
x_pos = np.array([-3])
y_vec = np.array([-2])
y_pos = np.array([2])



'''
def check_vector_continuity(vector1, vector2):
    if np.dot(vector1, vector2) > 0:
        flipped = False
    else: #if np.dot(vector1, vector2) < 0:
        flipped = True
    #else:
        #raise TypeError("Not 1D segments")
    return flipped
'''
'''
x = np.dot([1,0], [-2,0])
print(x)

length = len(df)
flipped = []

for i in range(0,1):
    if i ==  0:
        pass
    else:
        vector1 = df['output'].iloc[i-1] -df['input'].iloc[i-1]
        vector1 = np.array([vector1, 0])
        vector2 = df['output'].iloc[i] - df['input'].iloc[i]
        vector2 = np.array([vector2, 0])
        x = check_vector_continuity(vector1, vector2)
        flipped.append(x)

print(flipped)
'''
i = 2
vector1 = df['output'].iloc[i-1] -df['input'].iloc[i-1]
print(vector1)
#vector1 = np.array([vector1])
vector2 = df['output'].iloc[i] - df['input'].iloc[i]
print(vector2)
#vector2 = np.array([vector2])


fig, ax = plt.subplots()

ax.quiver([vector1], [0], 0, scale =1, color="red")
ax.quiver([vector2], [0], 0, scale =1)

x = np.dot(np.array([vector2]), np.array([vector1]))

print("dot product", x)
ax.set_xlim([-10,10])
#plt.show()
'''
def check_vector_continuity(vector1, vector2):
    if np.dot(vector1, vector2) >= 0:
        flipped = True
    if np.dot(vector1, vector2) <=0:
        flipped = False
    else:
        raise TypeError(f"Not 1D segments on line{i}")
    return flipped
'''
def check_vector_flip(vector1, vector2, verbose= False):
    dot = np.dot(vector1, vector2)
    if verbose:
        print(f"{vector1} dot {vector2} equals {dot}")
    if dot >= 0:
        if verbose:
            print('Not Flipped')
        flipped = False
    else: #if np.dot(vector1, vector2) < 0:
        flipped = True
    #else:
        #raise TypeError("Not 1D segments")
    if verbose:
        print(flipped)
    return flipped

length = len(df)
flipped = []

for i in range(0,length):
    if i ==  0:
        x = False
        flipped.append(x)
    else:
        vector1 = df['output'].iloc[i-1] -df['input'].iloc[i-1]
        vector1 = np.array([vector1])
        vector2 = df['output'].iloc[i] - df['input'].iloc[i]
        vector2 = np.array([vector2])
        x = check_vector_flip(vector2, vector1)
        flipped.append(x)

print(flipped)

checked_results = pd.DataFrame(flipped)
checked_results.columns = ['flipped']
print(checked_results)

def compare_results(original, my_version):
    similarity= []
    for i in range(0, length):
        if original["flipped_to_previous"].iloc[i] == my_version["flipped"].iloc[i]:
            same = True
        else:
            same = "NOT SAME"
        similarity.append(same)
    return similarity

df["Similarity"] = compare_results(df, checked_results)

print(df.loc[df['Similarity'] == 'NOT SAME'])
#comparison_df = pd.DataFrame(compare_results(df, checked_results))
#comparison_df.columns = ["Similarity"]
#print(comparison_df.loc[comparison_df['Similarity'] == 'NOT SAME'])

dff = df.loc[df['Similarity'] == 'NOT SAME'] #df.query("Similarity ==' NOT SAME'")
print(dff)
indexes = dff.index.values

length = len(dff)
flipped = []

for i in indexes:
    if i ==  0:
        x = False
        flipped.append(x)
    else:
        vector1 = df['output'].iloc[i-1] -df['input'].iloc[i-1]
        vector1 = np.array([vector1])
        vector2 = df['output'].iloc[i] - df['input'].iloc[i]
        vector2 = np.array([vector2])
        x = check_vector_flip(vector2, vector1,True)
        flipped.append(x)

print(flipped)