from typing import Self
import torch

from ..model import Model
from ...data import Dataset

class SASRec(Model):
    def __init__(self, dataset: Dataset, config: dict) -> None:
        super(SASRec, self).__init__(dataset, config)