import numpy as np
import torch
import torch.nn as nn


class EASE(nn.Module):
    def __init__(self, adj_mat, device="cpu"):
        super(EASE, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.item_adj = (adj_mat.T @ adj_mat).to(device)

    def forward(self, lambda_):
        G = self.item_adj
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = torch.inverse(G)
        B = P / (-torch.diag(P))
        B[diagIndices] = 0
        rating = torch.mm(self.adj_mat, B)

        return rating
