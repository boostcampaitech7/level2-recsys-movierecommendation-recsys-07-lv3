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
    train_loss = 0.0
    start_time = time.time()
    update_count = 0

    for batch_idx, start_idx in enumerate(
        range(0, train_data.shape[0], args.batch_size)
    ):
        end_idx = min(start_idx + args.batch_size, train_data.shape[0])
        data = train_data[start_idx:end_idx]
        data = torch.FloatTensor(data.toarray()).to(device)
        optimizer.zero_grad()

        if args.model == "MultiVAE":
            if args.total_anneal_steps > 0:
                anneal = min(
                    args.anneal_cap, 1.0 * update_count / args.total_anneal_steps
                )
            else:
                anneal = args.anneal_cap

        # Forward pass and loss computation
        if args.model == "MultiVAE":
            recon_batch, mu, logvar = model(data)
            loss = loss_function_vae(recon_batch, data, mu, logvar, anneal)
        else:
            recon_batch = model(data)
            # TODO
            # loss =

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        update_count += 1

        # # Log progress
        # if batch_idx % args.log_interval == 0 and batch_idx > 0:
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
        #           'loss {:4.2f}'.format(
        #               epoch, batch_idx, len(range(0, train_data.shape[0], args.batch_size)),
        #               elapsed * 1000 / args.log_interval,
        #               train_loss / args.log_interval))
        #     start_time = time.time()

    # Return average training loss
    return train_loss / len(range(0, train_data.shape[0], args.batch_size))


def evaluate():
    # TODO
    pass


def evaluate_MultiVAE(
    model,
    train_data,
    validation_data,
    loss_function,
    device,
    top_k=[10],
    anneal_cap=0.2,
    total_anneal_steps=20000,
):
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
            train_tensor = torch.FloatTensor(train_row).to(device)
            val_tensor = torch.FloatTensor(val_row).to(device)

            # Annealing factor for KL divergence
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1.0 * update_count / total_anneal_steps)

            # Forward pass
            recon_batch, mu, logvar = model(train_tensor)

            # Compute loss (using validation data)
            loss = loss_function(recon_batch, val_tensor, mu, logvar, anneal)
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
