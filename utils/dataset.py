import torch
from torch_geometric.data import Data, Dataset, HeteroData
import pickle
import os


def add_self_loops_to_data(data):
    for node_type, features in data.x_dict.items():
        num_nodes = features.shape[0]

        self_loop_index = torch.tensor([range(num_nodes)] * 2, dtype=torch.long)

        data[node_type, 'self_loop', node_type].edge_index = self_loop_index


class Task_Data(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'map' or key == 'label' or key == 'time_map' or key == 'time_map_one_hot':
            return None
        else:
            return super().__cat_dim__(key, value)


class Task_Data_Hetero(HeteroData):
    def __iter__(self):
        pass

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'map' or key == 'label' or key == 'time_map' or key == 'time_map_one_hot':
            return None
        else:
            return super().__cat_dim__(key, value)


class NNgTLDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, f)
                            for f in os.listdir(data_dir)
                            if f.startswith('training_data_') and f.endswith('.pkl')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], 'rb') as f:
            data = pickle.load(f)
        return data
