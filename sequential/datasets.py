import random

import torch
from torch.utils.data import Dataset






class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        target_neg = []

        assert self.data_type in {"train", "valid","submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3,4, 5 ]
        # target [1, 2, 3, 4,5, 6]

        # valid [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

       

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
       

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
