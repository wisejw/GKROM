# model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Linear


# Graph Attention Network (GAT) model
class GAT(torch.nn.Module):
    def __init__(self, in_channels=300, out_channels1=128, out_channels2=512):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels1, add_self_loops=False)
        self.lin1 = Linear(-1, out_channels1)
        self.conv2 = GATConv(out_channels1, out_channels2, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels2)

    def forward(self, x, edge_index):
        # Apply first GAT layer and add linear transformation
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        # Apply second GAT layer and add linear transformation
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class MHGAT(torch.nn.Module):
    def __init__(self, in_channels=300, out_channels1=128, out_channels2=512, heads=4):
        super(MHGAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels1, heads=heads, concat=True, add_self_loops=False)
        self.lin1 = Linear(-1, out_channels1 * heads)
        self.conv2 = GATConv(out_channels1 * heads, out_channels2, heads=heads, concat=True, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels2 * heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


# Siamese Network
class SiameseNet(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNet, self).__init__()
        self.fc_layer = nn.Linear(input_dim, 64)
        self.relu_layer = nn.ReLU()
        self.classificaton_layer = nn.Linear(256, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x1, x2):
        c1 = self.relu_layer(self.fc_layer(x1))
        c2 = self.relu_layer(self.fc_layer(x2))
        diff = torch.sub(c1, c2)
        multiply = torch.mul(c1, c2)
        v = torch.cat((c1, c2, diff, multiply), 1)
        pred_prob = self.sigmoid_layer(self.classificaton_layer(v))
        return pred_prob


# Resource-Concept Prediction Network
class ResourceConceptPredictNet(nn.Module):
    def __init__(self, input_dim):
        super(ResourceConceptPredictNet, self).__init__()
        self.fc1_layer = nn.Linear(input_dim, 64)
        self.fc2_layer = nn.Linear(input_dim, 64)
        self.relu_layer = nn.ReLU()
        self.classificaton_layer = nn.Linear(128, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, resource, concept):
        resource = self.relu_layer(self.fc1_layer(resource))
        concept = self.relu_layer(self.fc2_layer(concept))
        v = torch.cat((resource, concept), 1)
        pred_prob = self.sigmoid_layer(self.classificaton_layer(v))
        return pred_prob


# Resource-Dependency Prediction Network
class ResourceDependencyPredictNet(nn.Module):
    def __init__(self, input_dim):
        super(ResourceDependencyPredictNet, self).__init__()
        self.fc_layer = nn.Linear(input_dim, 64)
        self.relu_layer = nn.ReLU()
        self.classificaton_layer = nn.Linear(256, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x1, x2):
        r1 = self.relu_layer(self.fc_layer(x1))
        r2 = self.relu_layer(self.fc_layer(x2))
        diff = torch.sub(r1, r2)
        multiply = torch.mul(r1, r2)
        v = torch.cat((r1, r2, diff, multiply), 1)
        pred_prob = self.sigmoid_layer(self.classificaton_layer(v))
        return pred_prob
