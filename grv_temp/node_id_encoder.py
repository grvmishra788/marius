import pandas as pd
from tqdm import tqdm

  
def update_edges(file1,file2):
    
    # read the two files
    edges = pd.read_csv(file1, header=None)
    num_edges = pd.read_csv(file2, header=None)
    
    # create a list to store the updated edges
    updated_edges = []
    
    offset = 0

    # ipdb.set_trace()

    # loop through the files
    for i in tqdm(range(len(num_edges))):
        graph_id = i
        num_edges_in_graph = num_edges.iloc[i,0]
        for j in range(offset, offset + num_edges_in_graph):
            node1 = edges.iloc[j,0]
            node2 = edges.iloc[j,1]
            updated_node1 = str(graph_id * (10**10) + node1).zfill(20)
            updated_node2 = str(graph_id * (10**10) + node2).zfill(20)
            updated_edge = [updated_node1, updated_node2]
            updated_edges.append(updated_edge)
        offset += num_edges_in_graph
            
    # convert the list to a dataframe
    updated_edges_df = pd.DataFrame(updated_edges)
    
    # write the updated edges to a file
    updated_edges_df.to_csv('updated_edges.csv', index=False,  header=None)
    
    print('Updated edges successfully written to updated_edges.csv')
    

if __name__ == "__main__":
    update_edges("/marius/temp/hiv/raw/edge.csv", "/marius/temp/hiv/raw/num-edge-list.csv")