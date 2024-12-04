import os
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, HeteroConv

from .dataset import NNgTLDataset

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(8, 3, kernel_size=1)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class NodeClassifier(nn.Module):
    def __init__(self, metadata):
        super(NodeClassifier, self).__init__()

        self.gnn1 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=128, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn2 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=256, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn3 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=512, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn4 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=256, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn5 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=128, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')

        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        for gnn in [self.gnn1, self.gnn2, self.gnn3, self.gnn4, self.gnn5]:
            x = gnn(x, edge_index)
            x['node'] = self.dropout(F.relu(x['node']))
            x['edge'] = self.dropout(F.relu(x['edge']))
            x['source'] = self.dropout(F.relu(x['source']))

        x['node'] = self.linear1(x['node'])
        x['node'] = F.relu(x['node'])
        x['node'] = self.linear2(x['node'])

        return x['node']


class BA_Predict(nn.Module):
    def __init__(self, metadata):
        super(BA_Predict, self).__init__()
        self.image_feature_extractor = ImageFeatureExtractor()

        # Freeze parameters of the ResNet model
        # for param in self.image_feature_extractor.resnet.parameters():
        #     param.requires_grad = False

        self.node_classifier = NodeClassifier(metadata)
        self.gnn1 = HeteroConv({
            # edge_type: SAGEConv((-1, -1), out_channels=64, aggr='mean')
            edge_type: GATConv((-1, -1), out_channels=64, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')

    def forward(self, data_map, x, edge_index, batch):
        image_features = self.image_feature_extractor(data_map)
        x = self.gnn1(x, edge_index)
        for node_type in x.keys():
            node_features = x[node_type]
            batch_vector = batch[node_type]
            image_features_for_nodes = torch.index_select(image_features, 0, batch_vector)
            new_node_features = torch.cat([node_features, image_features_for_nodes], dim=-1)
            x[node_type] = new_node_features

        out1 = self.node_classifier(x, edge_index)

        return out1


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# BA_Encoder
class BA_Encoder(nn.Module):
    def __init__(self, metadata):
        super(BA_Encoder, self).__init__()

        self.gnn1 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=64, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn2 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=128, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn3 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=128, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn4 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=128, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.gnn5 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels=64, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch):
        x1 = self.gnn1(x, edge_index)
        x1['node'] = self.dropout(F.relu(x1['node']))
        x1['edge'] = self.dropout(F.relu(x1['edge']))
        x1['source'] = self.dropout(F.relu(x1['source']))
        x2 = self.gnn2(x1, edge_index)
        x2['node'] = self.dropout(F.relu(x2['node']))
        x2['edge'] = self.dropout(F.relu(x2['edge']))
        x2['source'] = self.dropout(F.relu(x2['source']))
        x3 = self.gnn3(x2, edge_index)
        x3['node'] = self.dropout(F.relu(x3['node']))
        x3['edge'] = self.dropout(F.relu(x3['edge']))
        x3['source'] = self.dropout(F.relu(x3['source']))
        x4 = self.gnn4(x3, edge_index)
        x4['node'] = self.dropout(F.relu(x4['node']))
        x4['edge'] = self.dropout(F.relu(x4['edge']))
        x4['source'] = self.dropout(F.relu(x4['source']))
        x5 = self.gnn5(x4, edge_index)
        x5['node'] = self.dropout(F.relu(x5['node']))
        x5['edge'] = self.dropout(F.relu(x5['edge']))
        x5['source'] = self.dropout(F.relu(x5['source']))

        # Pooling to get global graph representation
        x1_pool = torch.cat((
            global_mean_pool(x1['node'], batch['node']),
            global_mean_pool(x1['edge'], batch['edge']),
            global_mean_pool(x1['source'], batch['source'])
        ), dim=1)
        x2_pool = torch.cat((
            global_mean_pool(x2['node'], batch['node']),
            global_mean_pool(x2['edge'], batch['edge']),
            global_mean_pool(x2['source'], batch['source'])
        ), dim=1)
        x3_pool = torch.cat((
            global_mean_pool(x3['node'], batch['node']),
            global_mean_pool(x3['edge'], batch['edge']),
            global_mean_pool(x3['source'], batch['source'])
        ), dim=1)
        x4_pool = torch.cat((
            global_mean_pool(x4['node'], batch['node']),
            global_mean_pool(x4['edge'], batch['edge']),
            global_mean_pool(x4['source'], batch['source'])
        ), dim=1)
        x5_pool = torch.cat((
            global_mean_pool(x5['node'], batch['node']),
            global_mean_pool(x5['edge'], batch['edge']),
            global_mean_pool(x5['source'], batch['source'])
        ), dim=1)

        return x1_pool, x2_pool, x3_pool, x4_pool, x5_pool


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=True, output_padding=0):
        super(up_conv, self).__init__()
        if dropout:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, output_padding=output_padding,
                                   bias=True),
                nn.BatchNorm2d(out_ch),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, output_padding=output_padding,
                                   bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class Map_Encoder(nn.Module):
    def __init__(self, in_ch=8, n1=64):
        super(Map_Encoder, self).__init__()
        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Conv1 = conv_block(in_ch, self.filters[0])
        self.Conv2 = conv_block(self.filters[0], self.filters[1])
        self.Conv3 = conv_block(self.filters[1], self.filters[2])
        self.Conv4 = conv_block(self.filters[2], self.filters[3])
        self.Conv5 = conv_block(self.filters[3], self.filters[4])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        return x1, x2, x3, x4, x5


class Fusion_Net(nn.Module):
    def __init__(self, n1=64):
        super(Fusion_Net, self).__init__()
        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # Projection layers for GNN features to match dimensions of image encoder outputs
        self.project_gnn1 = nn.Linear(64 * 3, self.filters[0] * 100 * 100)
        self.project_gnn2 = nn.Linear(128 * 3, self.filters[1] * 50 * 50)
        self.project_gnn3 = nn.Linear(128 * 3, self.filters[2] * 25 * 25)
        self.project_gnn4 = nn.Linear(128 * 3, self.filters[3] * 12 * 12)
        self.project_gnn5 = nn.Linear(64 * 3, self.filters[4] * 6 * 6)

    def forward(self, image_features, gnn_features):
        # Project GNN features to match dimensions of image encoder outputs
        projected_gnn1 = self.project_gnn1(gnn_features[0]).view(-1, self.filters[0], 100, 100)
        projected_gnn2 = self.project_gnn2(gnn_features[1]).view(-1, self.filters[1], 50, 50)
        projected_gnn3 = self.project_gnn3(gnn_features[2]).view(-1, self.filters[2], 25, 25)
        projected_gnn4 = self.project_gnn4(gnn_features[3]).view(-1, self.filters[3], 12, 12)
        projected_gnn5 = self.project_gnn5(gnn_features[4]).view(-1, self.filters[4], 6, 6)

        # Concatenate features along the feature dimension
        fused_features1 = torch.cat((image_features[0], projected_gnn1), dim=1)
        fused_features2 = torch.cat((image_features[1], projected_gnn2), dim=1)
        fused_features3 = torch.cat((image_features[2], projected_gnn3), dim=1)
        fused_features4 = torch.cat((image_features[3], projected_gnn4), dim=1)
        fused_features5 = torch.cat((image_features[4], projected_gnn5), dim=1)

        return fused_features1, fused_features2, fused_features3, fused_features4, fused_features5


class Path_Predict(nn.Module):
    def __init__(self, out_ch=1, n1=64):
        super(Path_Predict, self).__init__()
        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Up5 = up_conv(self.filters[5], self.filters[4])
        self.Up4 = up_conv(self.filters[4], self.filters[3], dropout=True, output_padding=1)
        self.Up3 = up_conv(self.filters[3], self.filters[2], dropout=True)
        self.Up2 = up_conv(self.filters[2], self.filters[1], dropout=True)
        self.Up1 = up_conv(self.filters[1], self.filters[0], dropout=True)

        self.Conv6 = nn.Conv2d(self.filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.Up5(x5)
        # x = self.Up4(torch.cat((x, x4), dim=1))
        # x = self.Up3(torch.cat((x, x3), dim=1))
        # x = self.Up2(torch.cat((x, x2), dim=1))
        # x = self.Up1(torch.cat((x, x1), dim=1))
        x = self.Up4(x)
        x = self.Up3(x)
        x = self.Up2(x)
        x = self.Up1(x)
        x = self.Conv6(x)
        return x


class LTL_Net(nn.Module):
    def __init__(self, metadata, n1=64):
        super(LTL_Net, self).__init__()
        self.ba_encoder = BA_Encoder(metadata)
        self.map_encoder = Map_Encoder(n1=n1)
        self.fusion_net = Fusion_Net(n1=n1)
        self.path_predict = Path_Predict(n1=n1)

    def forward(self, data_map, x, edge_index_dict, batch):
        ba_features = self.ba_encoder(x, edge_index_dict, batch)
        image_features = self.map_encoder(data_map)
        fused_features = self.fusion_net(image_features, ba_features)
        out = self.path_predict(*fused_features)
        # out = F.sigmoid(out)
        out = out.squeeze(1)
        return out
