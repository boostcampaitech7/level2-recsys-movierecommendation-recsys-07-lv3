import torch

from ...data import Dataset
from .EASE import EASE
from ..trainer import Trainer

class EASETrainer(Trainer):
    def __init__(self, model: EASE, dataset: Dataset, config: dict):
        super(EASETrainer, self).__init__(model, dataset, config)
        self.model = model

    def train(self) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.calculate_rating_matrix()
        self.validate()

