from typing import Self
import numpy as np
import pandas as pd
import torch

from ..model import Model
from ...data import Dataset

class EASE(Model):
    def __init__(self, dataset: Dataset, config: dict) -> None:
        super(EASE, self).__init__(dataset, config)
        self.adj_matrix = torch.tensor(self.dataset.adj_matrix.todense(), dtype=torch.float)
    
    def calculate_rating_matrix(self) -> None:
        diag_indices = np.diag_indices(self.dataset.item_cnt)
        gram_matrix = self.adj_matrix.T @ self.adj_matrix
        gram_matrix[diag_indices] += self.config['lambda']
        inverse = torch.inverse(gram_matrix)
        inverse = inverse / (-torch.diag(inverse))
        inverse[diag_indices] = 0
        self.rating_matrix = self.adj_matrix @ inverse

    def get_topk(self, k: int) -> torch.Tensor:
        scores = self.rating_matrix
        trues = self.dataset.train_interactions.groupby('user_id')['item_id'].apply(list)
        for user_id, items in trues.items():
            scores[user_id, items] = -float('inf')
        topk = torch.topk(scores, k=k, dim=1).indices
        return topk
    
    def to(self, device: str) -> Self:
        super(EASE, self).to(device)
        self.adj_matrix = self.adj_matrix.to(device)
        return self
