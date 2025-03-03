import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix

from ..data import Dataset

class NegativeSampler:
    """Negative Sampling을 수행하는 클래스입니다.
    """

    def __init__(self, dataset: Dataset, sample_num_per_user: int, negative_sample_num: int) -> None:
        """Sampler의 설정을 받고 초기화합니다.

        Parameters
        ----------
        dataset : Dataset
            샘플을 추출할 데이터셋입니다.
        sample_num_per_user : int
            한 사용자 당 추출할 샘플의 수입니다.
        negative_sample_num : int
            한 positive 샘플 당 추출할 negative 샘플의 수입니다.
        """
        self.dataset = dataset
        self.sample_num_per_user = sample_num_per_user
        self.negative_sample_num = negative_sample_num

    def get_samples(self) -> torch.Tensor:
        """Negative Sampling을 수행하여 샘플을 반환합니다.

        Returns
        -------
        torch.Tensor
            Negative Sampling된 샘플입니다.
        """
        pairwise_samples = []

        for user in tqdm(range(self.dataset.user_cnt)):
            adj: csr_matrix = self.dataset.user_item_matrix
            all_items = np.arange(self.dataset.item_cnt)
            positive_items = adj.indices[adj.indptr[user]: adj.indptr[user + 1]]
            negative_items = np.setdiff1d(all_items, positive_items)

            for _ in range(self.sample_num_per_user):
                cur_positive_item = np.random.choice(positive_items)
                cur_negative_items = np.random.choice(negative_items, size=self.negative_sample_num)
                pairwise_samples.append([user, cur_positive_item, *cur_negative_items])
        return torch.tensor(pairwise_samples, dtype=torch.long)
