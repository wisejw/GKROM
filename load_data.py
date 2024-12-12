# load_data.py

import numpy as np
import torch
from torch_geometric.data import HeteroData


def coo_format(arr):
    value = arr[arr != 0]
    value = value[:, None]
    idx = np.argwhere(arr != 0)
    coo = idx.T
    return torch.tensor(value), torch.tensor(coo)


def load_data(device):
    df = np.loadtxt("feature/MOOC/df.txt")
    cf = np.loadtxt("feature/MOOC/cf.txt")
    dcf = np.loadtxt("feature/MOOC/dcf.txt")
    ddf = np.loadtxt("feature/MOOC/ddf.txt")
    ccf = np.loadtxt("feature/MOOC/ccf.txt")

    data = HeteroData()
    data['concept'].x = torch.tensor(cf)
    data['concept'].num_nodes = cf.shape[0]
    data['document'].x = torch.tensor(df)
    data['document'].num_nodes = df.shape[0]
    dcf_value, coo_dcf = coo_format(dcf)
    ddf_value, coo_ddf = coo_format(ddf)
    ccf_value, coo_ccf = coo_format(ccf)
    data['document', 'contains', 'concept'].edge_index = coo_dcf
    data['document', 'related_to', 'document'].edge_index = coo_ddf
    data['concept', 'similar_to', 'concept'].edge_index = coo_ccf
    data['document', 'contains', 'concept'].edge_attr = dcf_value
    data['document', 'related_to', 'document'].edge_attr = ddf_value
    data['concept', 'similar_to', 'concept'].edge_attr = ccf_value
    print(data)
    return data.to(device), data.to_homogeneous().to(device)
