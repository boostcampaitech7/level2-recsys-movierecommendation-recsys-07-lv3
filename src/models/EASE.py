import numpy as np
import torch
import torch.nn as nn

from src.data.preprocess import convert_sp_mat_to_sp_tensor


class EASE(nn.Module):
    def __init__(self, lambda_):
        super(EASE, self).__init__()
        self.lambda_ = lambda_
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, adj_mat):
        adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()
        adj_mat = adj_mat.to(self.device)
        item_adj = (adj_mat.T @ adj_mat).to(self.device)

        G = item_adj
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.lambda_
        P = torch.inverse(G)
        B = P / (-torch.diag(P))
        B[diagIndices] = 0
        rating = torch.mm(adj_mat, B)

        return rating
