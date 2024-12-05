import pathlib
import pandas as pd
import torch

from .data import Loader, Encoder, Splitter
from .data import Dataset
from .models import Model
from .models import Trainer
from . import models

class Manager:
    """모델을 학습하고 테스트하는 매니저 클래스입니다.
    """

    def __init__(self, dataset_config: dict, model_config: dict) -> None:
        """학습하는데에 있어서 필요한 설정을 받아 초기화합니다.

        Parameters
        ----------
        dataset_config : dict
            데이터셋의 종류와 경로, 그리고 검증 방식을 담은 딕셔너리입니다.
        model_config : dict
            모델의 종류와 하이퍼파라미터를 담은 딕셔너리입니다.
        """
        self.dataset_config = dataset_config
        self.model_config = model_config

    def train(self) -> None:
        """데이터를 불러오고 모델을 학습합니다.
        """
        loader = Loader()
        load_func = getattr(loader, f'load_{self.dataset_config["type"]}')
        splitter = Splitter()
        split_func = getattr(splitter, f'{self.dataset_config["split"]}_split')
        encoder = Encoder()
        path = pathlib.Path(self.dataset_config['path'])

        # 데이터를 불러오고 훈련 데이터와 검증 데이터로 나눈 뒤 인코딩합니다.
        interactions, user_info, item_info = load_func(path)
        train_interactions, test_interactions = split_func(interactions)
        train_interactions = encoder.fit_transform(train_interactions)
        test_interactions = encoder.transform(test_interactions)
        dataset = Dataset(train_interactions, test_interactions, user_info, item_info)

        # 모델을 학습합니다.
        model_cls = getattr(models, self.model_config['name'])
        model: Model = model_cls(dataset, self.model_config)
        trainer_cls = getattr(models, self.model_config['name'] + 'Trainer')
        trainer: Trainer = trainer_cls(model, dataset, self.model_config)
        trainer.train()

    def test(self) -> None:
        """데이터를 불러오고 모델을 테스트합니다.

        그 후, topk 예측을 생성하고 저장합니다.
        """
        loader = Loader()
        load_func = getattr(loader, f'load_{self.dataset_config["type"]}')
        encoder = Encoder()
        path = pathlib.Path(self.dataset_config['path'])

        # 데이터를 불러오고 인코딩합니다.
        interactions, user_info, item_info = load_func(path)
        interactions = encoder.fit_transform(interactions)
        dataset = Dataset(interactions, None, user_info, item_info)

        # 모델을 학습합니다.
        model_cls = getattr(models, self.model_config['name'])
        model: Model = model_cls(dataset, self.model_config)
        trainer_cls = getattr(models, self.model_config['name'] + 'Trainer')
        trainer: Trainer = trainer_cls(model, dataset, self.model_config)
        trainer.train()

        # 모델의 topk 예측을 생성합니다.
        model.eval()
        with torch.no_grad():
            topk = model.get_topk(10)
        model.train()
        
        # topk 예측을 디코딩하고 저장합니다.
        test_user_ids, test_item_ids = [], []
        for user_id, item_ids in enumerate(topk):
            test_user_ids.extend([user_id] * len(item_ids))
            test_item_ids.extend(item_ids.tolist())
        topk_df = pd.DataFrame({'user_id': test_user_ids, 'item_id': test_item_ids})
        topk_df = encoder.inverse_transform(topk_df)
        topk_df.columns = ['user', 'item']
        topk_df.to_csv('topk.csv', index=False)
            
        


    


            