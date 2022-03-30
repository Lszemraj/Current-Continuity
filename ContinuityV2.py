import numpy as np
import pickle as pkl
from funcs_file import make_conductor_dicts, load_all_geoms

df_dict = load_all_geoms()
generate_dict, conductor_dict, id_column_dict = make_conductor_dicts(df_dict)




# these tell us which index corresponds to which conductor key
sorted_dict = sorted(conductor_dict.items())
print(sorted_dict)
index_dict = {i:k[0] for i,k in enumerate(sorted_dict)} # save this
print(index_dict) #First num is index in matrix, second num is conductor index


datadir = '/home/shared_data/helicalc_params/'
min_dist_file = datadir+'min_dist_matrix.pkl'
connected_file = datadir+'connected_matrix.pkl'
close_file = datadir+'close_matrix.pkl'

with open(min_dist_file, 'rb') as file:
    min_dist_matrix = pkl.load(file)
# correct the 0 along the diagonal

min_dist_matrix = min_dist_matrix + 1000*np.identity(len(min_dist_matrix))

with open(connected_file, 'rb') as file:
    connected_matrix = pkl.load(file)
    connected_array = np.array(connected_matrix)

with open(close_file, 'rb') as file:
    close_matrix = pkl.load(file)

print(connected_matrix)
print(connected_matrix[0])
print("loc", np.argwhere(connected_array[0]))
print(connected_array[0][40])
print(np.amin(min_dist_matrix[0]))
print(np.where(min_dist_matrix[0] == np.amin(min_dist_matrix[0]) ))
connection_dict = {}

def find_connection(index_dict, min_dist_matrix, connected_array):
    for key, value in enumerate(index_dict):
        index_number = key
        conductor_num = value
        x = int(np.argwhere(connected_matrix[index_number]))
        if x:
            connected_conductor = index_dict[x]
            connection_dict[value] = connected_conductor
        else:
            closest = np.where(min_dist_matrix[key] == np.amin(min_dist_matrix[key]) )
            closest_conductor = index_dict[int(closest)]
            connection_dict[value] = closest_conductor


find_connection(index_dict, min_dist_matrix, connected_matrix)
print(connection_dict)






#print("minimum distance natrix", min_dist_matrix)
#print("connected matrix", connected_matrix)
#print("close_matrix", close_matrix)

connections = np.sum(connected_matrix, axis = 0)
#print(connections)
#print(len(connections))

connections2 = np.sum(connected_matrix, axis = 1)
#print(connections2)
#print(len(connections2))

unconnected_coils = []
#unconnected_coils_output_input = []

for i in range(0, len(connections)):
    if connections[i] == 0:
        #unconnected_coils_output_input.append(output_input[i])
        unconnected_coils.append(i+1)

#print(unconnected_coils)
output_input = np.argwhere(connections)
#print(output_input)

#def put_in_order(min_dist, coil_list):
