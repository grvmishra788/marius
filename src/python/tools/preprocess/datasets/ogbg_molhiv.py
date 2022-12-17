from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import ipdb
from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
import sys
sys.path.append('/marius/src/python/')
from helper_utils import *
sys.path.append('/marius/src/python/tools/preprocess/converters/')
from torch_converter import TorchEdgeListConverter
# sys.path is a list of absolute path strings
# sys.path.append('/marius/src/python/tools/preprocess/')
# from dataset import GraphClassificationDataset
from marius.tools.preprocess.dataset import GraphClassificationDataset
from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes
from marius.tools.preprocess.utils import download_url, extract_file


class OGBG_MOLHIV(GraphClassificationDataset):


    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbg_molhiv"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"

    def download(self, overwrite=False):
        self.input_edge_list_file = self.output_directory / Path("edge.csv")
        self.input_node_feature_file = self.output_directory / Path("node-feat.csv")
        self.input_graph_labels_file = self.output_directory / Path("graph-label.csv")
        self.input_train_nodes_file = self.output_directory / Path("train.csv")
        self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
        self.input_test_nodes_file = self.output_directory / Path("test.csv")

        # extra
        self.input_edge_feature_file = self.output_directory / Path("edge-feat.csv")
        self.input_num_edges_file = self.output_directory / Path("num-edge-list.csv")
        self.input_num_nodes_file = self.output_directory / Path("num-node-list.csv")
        

        download = False
        if not self.input_edge_list_file.exists():
            download = True
        if not self.input_node_feature_file.exists():
            download = True
        if not self.input_graph_labels_file.exists():
            download = True
        if not self.input_train_nodes_file.exists():
            download = True
        if not self.input_valid_nodes_file.exists():
            download = True
        if not self.input_test_nodes_file.exists():
            download = True

        # extra
        if not self.input_edge_feature_file.exists():
            download = True
        if not self.input_num_edges_file.exists():
            download = True
        if not self.input_num_nodes_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            extract_file(self.output_directory / Path("hiv/raw/edge.csv.gz"))
            extract_file(self.output_directory / Path("hiv/raw/node-feat.csv.gz"))
            extract_file(self.output_directory / Path("hiv/raw/graph-label.csv.gz"))
            extract_file(self.output_directory / Path("hiv/raw/edge-feat.csv.gz"))
            extract_file(self.output_directory / Path("hiv/raw/num-edge-list.csv.gz"))
            extract_file(self.output_directory / Path("hiv/raw/num-node-list.csv.gz"))

            (self.output_directory / Path("hiv/raw/edge.csv")).rename(self.input_edge_list_file)
            (self.output_directory / Path("hiv/raw/node-feat.csv")).rename(self.input_node_feature_file)
            (self.output_directory / Path("hiv/raw/graph-label.csv")).rename(self.input_graph_labels_file)
            (self.output_directory / Path("hiv/raw/edge-feat.csv")).rename(self.input_edge_feature_file)
            (self.output_directory / Path("hiv/raw/num-edge-list.csv")).rename(self.input_num_edges_file)
            (self.output_directory / Path("hiv/raw/num-node-list.csv")).rename(self.input_num_nodes_file)

            for file in (self.output_directory / Path("hiv/split/scaffold")).iterdir():
                extract_file(file)

            for file in (self.output_directory / Path("hiv/split/scaffold")).iterdir():
                file.rename(self.output_directory / Path(file.name))

            update_edges(self.input_edge_list_file, self.input_num_edges_file)

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):

        # replicate graph labels by number of nodes present in num-node-list.csv  & update graph-label.csv
        gl = np.genfromtxt(self.input_graph_labels_file, delimiter=",").astype(np.int32)
        if gl.shape[0] != 1049163:
            num_nodes_g = np.genfromtxt(self.input_num_nodes_file, delimiter=",").astype(np.int32)
            gl = np.repeat(gl, num_nodes_g).astype(np.int32)
            np.savetxt(self.input_graph_labels_file, gl, delimiter=",")
            # print(gl.shape)


        train_nodes = np.genfromtxt(self.input_train_nodes_file, delimiter=",").astype(np.int32)
        valid_nodes = np.genfromtxt(self.input_valid_nodes_file, delimiter=",").astype(np.int32)
        test_nodes = np.genfromtxt(self.input_test_nodes_file, delimiter=",").astype(np.int32)
        # get updated train, valid, test nodes
        train_nodes = get_updated_nodes(train_nodes)
        valid_nodes = get_updated_nodes(valid_nodes)
        test_nodes = get_updated_nodes(test_nodes)
        # print(train_nodes.shape, valid_nodes.shape, test_nodes.shape)

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_edge_list_file,
            num_partitions=num_partitions,
            columns=[0, 1],
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            delim=",",
            known_node_ids=[train_nodes, valid_nodes, test_nodes],
            partitioned_evaluation=partitioned_eval,
            num_nodes=1049163, # TODO: USe dynamic
            num_rels=1 # TODO: USe dynamic
        )
        dataset_stats = converter.convert()

        features = np.genfromtxt(self.input_node_feature_file, delimiter=",").astype(np.float32)
        labels = np.genfromtxt(self.input_graph_labels_file, delimiter=",").astype(np.int32)
        # ipdb.set_trace()
        # extras 
        edge_features = np.genfromtxt(self.input_edge_feature_file, delimiter=",").astype(np.float32)
        num_edges = np.genfromtxt(self.input_num_edges_file, delimiter=",").astype(np.int32)
        num_nodes = np.genfromtxt(self.input_num_nodes_file, delimiter=",").astype(np.int32)

        # if remap_ids: # TODO: Add edge features
        #     node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            # train_nodes, valid_nodes, test_nodes, features, labels = remap_nodes(
            #     node_mapping, train_nodes, valid_nodes, test_nodes, features, labels
            # )

        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.graph_labels_file, "wb") as f:
            f.write(bytes(labels))

        # extra
        with open(self.relation_features_file, "wb") as f:
            f.write(bytes(edge_features))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.node_feature_dim = features.shape[1]
        dataset_stats.num_classes = 40

        dataset_stats.num_nodes = dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_test
        # dataset_stats.num_nodes = 1049163

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats
