import torch

from ...data import Dataset
from .SVDAE import SVDAE
from ..trainer import Trainer

class SVDAETrainer(Trainer):
    def __init__(self, model: SVDAE, dataset: Dataset, config: dict):
        super(SVDAETrainer, self).__init__(model, dataset, config)
        self.model = model

    def train(self) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.calculate_rating_matrix()
        self.validate()
