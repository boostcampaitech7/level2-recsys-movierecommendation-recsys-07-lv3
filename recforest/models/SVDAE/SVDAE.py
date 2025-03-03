from typing import Self
import torch

from ..model import Model
from ...data import Dataset

class SVDAE(Model):
    def __init__(self, dataset: Dataset, config: dict) -> None:
        super(SVDAE, self).__init__(dataset, config)
        self.adj_matrix = torch.tensor(self.dataset.adj_matrix.todense(), dtype=torch.float)

    def get_topk(self, k: int) -> torch.Tensor:
        scores = self.rating_matrix
        trues = self.dataset.train_interactions.groupby('user_id')['item_id'].apply(list)
        for user_id, items in trues.items():
            scores[user_id, items] = -float('inf')
        topk = torch.topk(scores, k=k, dim=1).indices
        return topk
    
    def calculate_rating_matrix(self) -> None:
        normalized_adj_matrix = self.get_normalized_adj_matrix()
        k = int(self.config['lambda'] * min(self.dataset.user_cnt, self.dataset.item_cnt))
        u, s, v = torch.svd_lowrank(normalized_adj_matrix, q=k)
        recovered_matrix = v @ torch.diag(1 / s) @ u.T
        self.rating_matrix = normalized_adj_matrix @ recovered_matrix @ self.adj_matrix
    
    def get_normalized_adj_matrix(self) -> torch.Tensor:
        row_sum = self.adj_matrix.sum(axis=1).flatten()
        row_sum[row_sum == 0] = 1.0
        d_row_inv_sqrt = torch.pow(row_sum, -0.5)
        d_row_inv_sqrt_mat = torch.diag(d_row_inv_sqrt)
        row_normalized = d_row_inv_sqrt_mat @ self.adj_matrix

        col_sum = self.adj_matrix.sum(axis=0).flatten()
        col_sum[col_sum == 0] = 1.0
        d_col_inv_sqrt = torch.pow(col_sum, -0.5)
        d_col_inv_sqrt_mat = torch.diag(d_col_inv_sqrt)
        normalized_adj_matrix = row_normalized @ d_col_inv_sqrt_mat
        return normalized_adj_matrix
    
    def to(self, device: str) -> Self:
        super(SVDAE, self).to(device)
        self.adj_matrix = self.adj_matrix.to(device)
        return self
