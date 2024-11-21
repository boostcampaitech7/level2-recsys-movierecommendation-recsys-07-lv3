import time
import numpy as np
import torch

from src.train.loss import loss_function_vae
from src.train.metric import Recall_at_k_batch, NDCG_binary_at_k_batch


def train(model, criterion, optimizer, train_data, device, args):
    """
    Train the model on the given training data.

    Parameters:
    - model: PyTorch model
    - criterion: Loss function
    - optimizer: Optimizer for training
    - train_data: csr_matrix, training data
    - device: PyTorch device (e.g., 'cuda' or 'cpu')
    - args: Arguments containing hyperparameters

    Returns:
    - train_loss: Average training loss over all batches
    """
    # Turn on training mode
    model.train()
    model.to(device)
    num_users = train_data.shape[0]
    batch_size = args.train.batch_size
    train_loss = 0.0
    # start_time = time.time()
    update_count = 0

    for batch_idx, start_idx in enumerate(range(0, num_users, batch_size)):
        end_idx = min(start_idx + batch_size, num_users)
        data = train_data[start_idx:end_idx]
        data = torch.FloatTensor(data.toarray()).to(device)
        optimizer.zero_grad()

        if args.model == "MultiVAE":
            # load config hyperparams
            total_anneal_steps = args.total_anneal_steps
            anneal_cap = args.anneal_cap

            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1.0 * update_count / total_anneal_steps)
            else:
                # anneal is constantly at anneal_cap
                anneal = anneal_cap

        # Forward pass and loss computation
        if args.model == "MultiVAE":
            recon_batch, mu, logvar = model(data)
            loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:  # Other models
            recon_batch = model(data)
            loss = criterion(recon_batch, data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        update_count += 1

    # Return average training loss
    return train_loss / update_count

    # # Log progress
    # if batch_idx % args.log_interval == 0 and batch_idx > 0:
    #     elapsed = time.time() - start_time
    #     print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
    #           'loss {:4.2f}'.format(
    #               epoch, batch_idx, update_count,
    #               elapsed * 1000 / args.log_interval,
    #               train_loss / args.log_interval))
    #     start_time = time.time()


# import torch
# from torch.utils.data import Dataset, DataLoader
# from scipy.sparse import csr_matrix


# class SparseDataset(Dataset):
#     def __init__(self, csr_data):
#         self.data = csr_data

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         return self.data[idx]


# def sparse_collate_fn(batch):
#     samples = len(batch)
#     features = batch[0].shape[1]

#     indices = []
#     values = []

#     for i, sample in enumerate(batch):
#         coo = sample.tocoo()
#         indices.append(torch.LongTensor([coo.row + i * sample.shape[0], coo.col]))
#         values.append(torch.FloatTensor(coo.data))

#     indices = torch.cat(indices, dim=1)
#     values = torch.cat(values)

#     return torch.sparse.FloatTensor(
#         indices, values, [samples * batch[0].shape[0], features]
#     )


# def train(model, criterion, optimizer, train_data, device, args):
#     """
#     Train the model on the given training data using DataLoader.

#     Parameters:
#     - model: PyTorch model
#     - criterion: Loss function
#     - optimizer: Optimizer for training
#     - train_data: csr_matrix, training data
#     - device: PyTorch device (e.g., 'cuda' or 'cpu')
#     - args: Arguments containing hyperparameters

#     Returns:
#     - train_loss: Average training loss over all batches
#     """
#     model.train()
#     model.to(device)
#     train_loss = 0.0
#     update_count = 0

#     # Create dataset and dataloader
#     dataset = SparseDataset(train_data)
#     dataloader = DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sparse_collate_fn
#     )

#     for batch_idx, data in enumerate(dataloader):
#         data = data.to(device)
#         optimizer.zero_grad()

#         if args.model == "MultiVAE":
#             total_anneal_steps = args.MultiVAE.total_anneal_steps
#             anneal_cap = args.MultiVAE.anneal_cap

#             if total_anneal_steps > 0:
#                 anneal = min(anneal_cap, 1.0 * update_count / total_anneal_steps)
#             else:
#                 anneal = anneal_cap

#             recon_batch, mu, logvar = model(data)
#             loss = criterion(recon_batch, data, mu, logvar, anneal)
#         else:  # Other models
#             recon_batch = model(data)
#             loss = criterion(recon_batch, data)

#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         update_count += 1

#     return train_loss / update_count


def evaluate():
    # TODO
    pass


def evaluate_MultiVAE(model, train_data, validation_data, top_k, loss_function, args):
    """
    Evaluate the Multi-VAE model on validation data.

    Parameters:
    - model: Multi-VAE model instance
    - train_data: csr_matrix, user-item interactions in the training set
    - validation_data: csr_matrix, user-item interactions in the validation set
    - loss_function: Loss function for VAE (e.g., loss_function_vae)
    - device: PyTorch device (e.g., 'cuda' or 'cpu')
    - top_k: List of k values for Recall@k and NDCG@k
    - anneal_cap: Maximum annealing factor for KL divergence
    - total_anneal_steps: Total steps for annealing KL divergence weight

    Returns:
    - avg_loss: Average evaluation loss
    - metrics: Dictionary containing Recall@k and NDCG@k for each k
    """
    model.eval()
    total_loss = 0.0
    update_count = 0
    total_anneal_steps = args.total_anneal_steps
    anneal_cap = args.anneal_cap
    anneal = 0.0
    recall_results = {k: [] for k in top_k}
    ndcg_results = {k: [] for k in top_k}

    num_users = validation_data.shape[0]

    with torch.no_grad():
        for user in range(num_users):  # User-wise evaluation
            # Get training and validation data for the current user
            train_row = train_data[user].toarray()
            val_row = validation_data[user].toarray()

            # Convert to PyTorch tensors
            train_tensor = torch.FloatTensor(train_row).to(args.device)
            val_tensor = torch.FloatTensor(val_row).to(args.device)

            if args.model == "MultiVAE":
                # Annealing factor for KL divergence
                if total_anneal_steps > 0:
                    anneal = min(
                        anneal_cap,
                        1.0 * update_count / total_anneal_steps,
                    )

                # Forward pass
                recon_batch, mu, logvar = model(train_tensor)

                # Compute loss (using validation data)
                loss = loss_function(recon_batch, val_tensor, mu, logvar, anneal)
            else:
                # Forward pass
                recon_batch = model(train_tensor)

                # Compute loss (using validation data)
                loss = loss_function(recon_batch, val_tensor)

            total_loss += loss.item()

            # Exclude training interactions from recommendations
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[train_row.nonzero()] = -np.inf  # Mask training items

            # Compute metrics for top_k
            for k in top_k:
                recall_results[k].append(Recall_at_k_batch(recon_batch, val_row, k))
                ndcg_results[k].append(NDCG_binary_at_k_batch(recon_batch, val_row, k))

            update_count += 1

    # Compute average loss and metrics
    avg_loss = total_loss / num_users
    metrics = {f"Recall@{k}": np.mean(recall_results[k]) for k in top_k}
    metrics.update({f"NDCG@{k}": np.mean(ndcg_results[k]) for k in top_k})

    return avg_loss, metrics
