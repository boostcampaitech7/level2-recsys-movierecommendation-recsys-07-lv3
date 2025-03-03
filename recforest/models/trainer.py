from abc import abstractmethod
import pandas as pd
import torch

from ..data import Dataset
from ..models import Model
from ..utils.metric import recall

class Trainer:
    """모델을 학습하는 Trainer의 추상 클래스입니다.
    """

    def __init__(self, model: Model, dataset: Dataset, config: dict):
        """Trainer 클래스를 초기화합니다.

        Parameters
        ----------
        model : Model
            Trainer와 대응되는 모델 객체입니다.
        dataset : Dataset
            학습에 사용할 데이터셋 객체입니다.
        config : dict
            Trainer와 모델에 대한 설정을 담은 딕셔너리입니다.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def train(self) -> None:
        """모델을 학습합니다.
        """
        pass

    def validate(self):
        """검증 데이터에 대한 평가를 수행하고 결과를 출력합니다.
        """
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

