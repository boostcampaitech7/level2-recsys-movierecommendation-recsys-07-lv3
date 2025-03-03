import numpy as np
import pandas as pd
from tqdm import tqdm

class Splitter:

    def aistages_split(self, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        grouped = interactions.groupby('user_id')
        train_users, train_items = [], []
        val_users, val_items = [], []

        for user, group in tqdm(grouped):
            item_indices = group['item_id'].values
            assert group['timestamp'].max() == group['timestamp'].iloc[-1]
            last_item = item_indices[-1]
            random_items = np.random.choice(item_indices[:-1], replace=False, size=1)
            cur_val_items = [last_item, *random_items]
            cur_train_items = list(set(item_indices) - set(cur_val_items))

            train_users.extend([user] * len(cur_train_items))
            val_users.extend([user] * len(cur_val_items))
            train_items.extend(cur_train_items)
            val_items.extend(cur_val_items)

        train_interactions = pd.DataFrame({'user_id': train_users, 'item_id': train_items})
        val_interactions = pd.DataFrame({'user_id': val_users, 'item_id': val_items})
        return train_interactions, val_interactions
    
    def predefined_split(self, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_interactions = interactions[interactions['train_or_test'] == 'train']
        test_interactions = interactions[interactions['train_or_test'] == 'test']
        return train_interactions, test_interactions

