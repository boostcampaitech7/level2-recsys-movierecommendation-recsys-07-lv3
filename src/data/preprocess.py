import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from typing import Tuple, Dict


# reindexing function
def reindex_column(
    data: pd.DataFrame, column_name: str
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Reindex a column in the dataframe to ensure continuous indices starting from 0.

    Parameters:
    - data: pd.DataFrame, the input dataframe.
    - column_name: str, the column to reindex.

    Returns:
    - data: pd.DataFrame, the dataframe with reindexed column.
    - mapping_dict: dict, the original-to-new mapping dictionary.
    """
    # Create the mapping dictionary
    mapping_dict = {
        original_id: new_id
        for new_id, original_id in enumerate(data[column_name].unique())
    }

    # Apply the mapping to the dataframe
    data[column_name] = data[column_name].map(mapping_dict)

    return data, mapping_dict


def sparse2torch_sparse(data: csr_matrix) -> torch.sparse.FloatTensor:
    """
    Convert scipy sparse matrix to torch sparse tensor
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    """
    row = data.shape[0]
    col = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    values = torch.FloatTensor(coo_data.data)

    return torch.sparse.FloatTensor(indices, values, [row, col])


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


# train/val split function
def train_test_split(
    interaction_matrix: csr_matrix, num_random_items: int = 2, sequence: bool = True
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Split a CSR interaction matrix into training and validation sets with specific rules:
    - Last interaction is always included in the validation set.
    - Additional two interactions are randomly selected.

    Parameters:
    - interaction_matrix: csr_matrix, the full user-item interaction matrix.

    Returns:
    - train_matrix: csr_matrix, training set interactions.
    - validation_matrix: csr_matrix, validation set interactions.
    """
    train_rows, train_cols, train_data = [], [], []
    val_rows, val_cols, val_data = [], [], []

    # Iterate over each user in the interaction matrix
    for user in range(interaction_matrix.shape[0]):
        # Get the non-zero interactions (item indices) for this user
        item_indices = interaction_matrix[user].nonzero()[1]
        num_items = len(item_indices)

        if num_items == 0:
            continue  # Skip users with no interactions

        # Last interaction added into the validation set
        if sequence:
            val_items = [item_indices[-1]]

        # Randomly sample additional two items from the remaining interactions
        random_items = np.random.choice(
            item_indices[:-1], size=num_random_items, replace=False
        )
        val_items.extend(random_items)

        # Add the remaining items to the training set
        train_items = list(set(item_indices) - set(val_items))

        # Add training interactions
        train_rows.extend([user] * len(train_items))
        train_cols.extend(train_items)
        train_data.extend([1] * len(train_items))

        # Add validation interactions
        val_rows.extend([user] * len(val_items))
        val_cols.extend(val_items)
        val_data.extend([1] * len(val_items))

    # Create CSR matrices for training and validation
    train_matrix = csr_matrix(
        (train_data, (train_rows, train_cols)),
        dtype="float64",
        shape=interaction_matrix.shape,
    )
    validation_matrix = csr_matrix(
        (val_data, (val_rows, val_cols)),
        dtype="float64",
        shape=interaction_matrix.shape,
    )

    return train_matrix, validation_matrix
