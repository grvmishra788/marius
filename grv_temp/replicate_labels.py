import pandas as pd

# Read in the data
df = pd.read_csv("/marius/temp/hiv/raw/graph-label.csv", header=None)
df2 = pd.read_csv("/marius/temp/hiv/raw/num-node-list.csv", header=None)

# Create a list to store the updated labels
updated_labels = []

# Loop through the files
for i in range(len(df2)):
    graph_id = i
    num_nodes_in_graph = df2.iloc[i,0]
    label = df.iloc[i,0]
    updated_labels.extend([label] * num_nodes_in_graph)

# Convert the list to a dataframe
updated_labels_df = pd.DataFrame(updated_labels)

# Write the updated labels to a file
updated_labels_df.to_csv("/marius/temp/hiv/raw/updated_labels.csv", index=False, header=None)

print("Updated labels successfully written to updated_labels.csv")
