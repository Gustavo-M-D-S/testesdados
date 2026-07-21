from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, GATConv, RGCNConv, Linear
from torch_geometric.data import HeteroData

from loguru import logger


# -------------------------
# BASE MODEL
# -------------------------

class BaseGNN(nn.Module):
    """
    Base class for all GNN models in BankGraphAI.
    """

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, data: HeteroData):
        raise NotImplementedError


# -------------------------
# GRAPH SAGE
# -------------------------

class GraphSAGEModel(BaseGNN):
    def __init__(self, metadata, hidden_channels: int, out_channels: int):
        super().__init__(hidden_channels, out_channels)

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict = {
            k: F.relu(self.conv1(x_dict[k], edge_index_dict[k]))
            for k in x_dict.keys()
        }

        x_dict = {
            k: self.conv2(x_dict[k], edge_index_dict[k])
            for k in x_dict.keys()
        }

        return x_dict


# -------------------------
# GAT MODEL
# -------------------------

class GATModel(BaseGNN):
    def __init__(self, metadata, hidden_channels: int, out_channels: int):
        super().__init__(hidden_channels, out_channels)

        self.conv1 = GATConv((-1, -1), hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=1)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict = {
            k: F.elu(self.conv1(x_dict[k], edge_index_dict[k]))
            for k in x_dict.keys()
        }

        x_dict = {
            k: self.conv2(x_dict[k], edge_index_dict[k])
            for k in x_dict.keys()
        }

        return x_dict


# -------------------------
# RGCN MODEL (HETEROGENEOUS)
# -------------------------

class RGCNModel(BaseGNN):
    def __init__(self, metadata, hidden_channels: int, out_channels: int):
        super().__init__(hidden_channels, out_channels)

        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations=len(self.edge_types))
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=len(self.edge_types))

        self.input_lin = nn.ModuleDict({
            ntype: Linear(-1, hidden_channels)
            for ntype in self.node_types
        })

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x = {}

        for k, v in x_dict.items():
            x[k] = F.relu(self.input_lin[k](v))

        # flatten edges
        edge_index_list = []
        edge_type_list = []

        for i, etype in enumerate(edge_index_dict.keys()):
            edge_index_list.append(edge_index_dict[etype])
            edge_type_list.append(torch.full((edge_index_dict[etype].size(1),), i))

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list)

        x_flat = torch.cat(list(x.values()), dim=0)

        x = F.relu(self.conv1(x_flat, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)

        return x


# -------------------------
# MODEL FACTORY
# -------------------------

@dataclass
class ModelConfig:
    model_name: str
    hidden_channels: int
    out_channels: int


class ModelFactory:
    """
    Creates GNN models dynamically from config.
    """

    @staticmethod
    def create(metadata, config: ModelConfig) -> BaseGNN:

        logger.info(f"Creating model: {config.model_name}")

        if config.model_name == "graphsage":
            return GraphSAGEModel(
                metadata,
                config.hidden_channels,
                config.out_channels,
            )

        if config.model_name == "gat":
            return GATModel(
                metadata,
                config.hidden_channels,
                config.out_channels,
            )

        if config.model_name == "rgcn":
            return RGCNModel(
                metadata,
                config.hidden_channels,
                config.out_channels,
            )

        raise ValueError(f"Unknown model: {config.model_name}")