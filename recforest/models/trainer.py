from abc import abstractmethod
import pandas as pd
import torch

from ..data import Dataset
from ..models import Model
from ..utils.metric import recall

class Trainer:

    def __init__(self, model: Model, dataset: Dataset, config: dict):
        self.model = model
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def train(self) -> None:
        pass

    def validate(self):
        if self.dataset.test_interactions is None:
            return
        self.model.eval()
        with torch.no_grad():
            pred = self.model.get_topk(10).to('cpu').numpy().tolist()
        self.model.train()
        grouped = self.dataset.test_interactions.groupby('user_id')['item_id'].apply(list)
        true = [grouped.get(user_id, []) for user_id in range(self.dataset.user_cnt)]
        metric = recall(true, pred, normalized=True)
        print(f'recall: {metric}')

