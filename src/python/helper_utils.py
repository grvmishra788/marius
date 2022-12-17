import numpy as np
import pandas as pd

def get_updated_nodes(nodes):
    updated = []
    num_nodes_g = np.genfromtxt("/marius/datasets/ogbg_molhiv/num-node-list.csv", delimiter=",").astype(np.int32)
    for graph in nodes:
        nn = num_nodes_g[graph]
        for i in range(nn):
            updated.append(graph*1000+i)
    return np.asarray(updated)


def update_edges(file1,file2):
    
    print("Updating edges")
    # read the two files
    edges = pd.read_csv(file1, header=None)
    num_edges = pd.read_csv(file2, header=None)
    
    # create a list to store the updated edges
    updated_edges = []
    
    offset = 0

    # ipdb.set_trace()

    # loop through the files
    for i in range(len(num_edges)):
        graph_id = i
        num_edges_in_graph = num_edges.iloc[i,0]
        for j in range(offset, offset + num_edges_in_graph):
            node1 = edges.iloc[j,0]
            node2 = edges.iloc[j,1]
            updated_node1 = graph_id * (1000) + node1
            updated_node2 = graph_id * (1000) + node2
            updated_edge = [updated_node1, updated_node2]
            updated_edges.append(updated_edge)
        offset += num_edges_in_graph
            
    # convert the list to a dataframe
    updated_edges_df = pd.DataFrame(updated_edges)
    
    # write the updated edges to a file
    updated_edges_df.to_csv(file1, index=False,  header=None)
    
    print(f'Updated edges successfully written to {file1}')