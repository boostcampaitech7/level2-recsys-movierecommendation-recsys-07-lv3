from abc import abstractmethod
import torch

from ..data import Dataset

class Model(torch.nn.Module):

    def __init__(self, dataset: Dataset, config: dict) -> None:
        super(Model, self).__init__()
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def get_topk(self, k: int) -> torch.Tensor:
        pass