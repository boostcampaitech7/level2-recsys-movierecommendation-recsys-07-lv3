import pathlib
import numpy as np
import pandas as pd

class Loader:

    def load_movielens_32m(self, path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions = pd.read_csv(path / 'ratings.csv')
        interactions.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        return interactions, None, None

    def load_gowalla(self, path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions = pd.read_csv(path / 'Gowalla_totalCheckins.txt', delimiter=r'\s+', header=None)
        interactions.columns = ['user_id', 'timestamp', 'latitude', 'longitude', 'item_id']
        return interactions, None, None
    
    def load_gowalla_lightgcn(self, path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with open(path / 'train.txt', 'r') as f:
            lines = f.readlines()
        lists = [list(map(int, line.split())) for line in lines]
        lists = [[list_[0], list_[i]] for list_ in lists for i in range(1, len(list_))]
        train_interactions = pd.DataFrame(lists, columns=['user_id', 'item_id'])
        train_interactions['train_or_test'] = 'train'

        with open(path / 'test.txt', 'r') as f:
            lines = f.readlines()
        lists = [list(map(int, line.split())) for line in lines]
        lists = [[list_[0], list_[i]] for list_ in lists for i in range(1, len(list_))]
        test_interactions = pd.DataFrame(lists, columns=['user_id', 'item_id'])
        test_interactions['train_or_test'] = 'test'

        interactions = pd.concat([train_interactions, test_interactions])     
        return interactions, None, None
    
    def load_aistages(self, path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions = pd.read_csv(path / 'train_ratings.csv')
        interactions.columns = ['user_id', 'item_id', 'timestamp']
        return interactions, None, None