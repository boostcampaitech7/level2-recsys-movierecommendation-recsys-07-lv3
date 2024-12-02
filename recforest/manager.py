import pathlib
import pandas as pd
import torch

from .data import Loader, Encoder, Splitter
from .data import Dataset
from .models import Model
from .models import Trainer
from . import models

class Manager:

    def __init__(self, dataset_config: dict, model_config: dict) -> None:
        self.dataset_config = dataset_config
        self.model_config = model_config

    def train(self) -> None:
        loader = Loader()
        load_func = getattr(loader, f'load_{self.dataset_config["type"]}')
        splitter = Splitter()
        split_func = getattr(splitter, f'{self.dataset_config["split"]}_split')
        encoder = Encoder()
        path = pathlib.Path(self.dataset_config['path'])

        interactions, user_info, item_info = load_func(path)
        train_interactions, test_interactions = split_func(interactions)
        train_interactions = encoder.fit_transform(train_interactions)
        test_interactions = encoder.transform(test_interactions)

        dataset = Dataset(train_interactions, test_interactions, user_info, item_info)

        model_cls = getattr(models, self.model_config['name'])
        model: Model = model_cls(dataset, self.model_config)
        trainer_cls = getattr(models, self.model_config['name'] + 'Trainer')
        trainer: Trainer = trainer_cls(model, dataset, self.model_config)
        trainer.train()

    def test(self) -> None:
        loader = Loader()
        load_func = getattr(loader, f'load_{self.dataset_config["type"]}')
        encoder = Encoder()
        path = pathlib.Path(self.dataset_config['path'])

        interactions, user_info, item_info = load_func(path)
        interactions = encoder.fit_transform(interactions)
        dataset = Dataset(interactions, None, user_info, item_info)

        model_cls = getattr(models, self.model_config['name'])
        model: Model = model_cls(dataset, self.model_config)
        trainer_cls = getattr(models, self.model_config['name'] + 'Trainer')
        trainer: Trainer = trainer_cls(model, dataset, self.model_config)
        trainer.train()

        model.eval()
        with torch.no_grad():
            topk = model.get_topk(10)
        model.train()
        
        test_user_ids, test_item_ids = [], []
        for user_id, item_ids in enumerate(topk):
            test_user_ids.extend([user_id] * len(item_ids))
            test_item_ids.extend(item_ids.tolist())
        topk_df = pd.DataFrame({'user_id': test_user_ids, 'item_id': test_item_ids})
        topk_df = encoder.inverse_transform(topk_df)
        topk_df.columns = ['user', 'item']
        
        topk_df.to_csv('topk.csv', index=False)
            
        


    


            