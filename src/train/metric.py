import numpy as np
import bottleneck as bn
from collections import defaultdict


def recall_at_k(ground_truth, predictions, k):
    """
    Compute Recall@K for each user.

    Args:
    - ground_truth: scipy.sparse.csr_matrix, the test set
    - predictions: numpy.ndarray, predicted scores for each user-item pair
    - k: int, the number of top items to consider

    Returns:
    - float, average Recall@K across all users
    """
    num_users = ground_truth.shape[0]

    # For each user, get the top-k item predictions
    top_k_items = np.argsort(-predictions, axis=1)[:, :k]

    recalls = []
    for user in range(num_users):
        user_ground_truth = ground_truth[user].indices
        user_top_k = top_k_items[user]

        # Compute recall for this user
        recall = len(set(user_top_k) & set(user_ground_truth)) / len(user_ground_truth)
        recalls.append(recall)

    return np.mean(recalls)


def recall_at_k_from_recommendations(recommendations, ground_truth, k):
    """
    Compute Recall@K using the recommendations list.

    Args:
    - recommendations: list of [user_id, item_id] pairs
    - ground_truth: scipy.sparse.csr_matrix, the test set
    - k: int, the number of top items to consider

    Returns:
    - float, average Recall@K across all users
    """
    user_recommendations = defaultdict(list)
    for user_id, item_id in recommendations:
        if len(user_recommendations[user_id]) < k:
            user_recommendations[user_id].append(item_id)

    recalls = []
    num_users = ground_truth.shape[0]

    for user in range(num_users):
        user_ground_truth = set(ground_truth[user].indices)
        user_top_k = set(user_recommendations.get(user, []))

        if len(user_ground_truth) > 0:
            recall = len(user_top_k & user_ground_truth) / len(user_ground_truth)
            recalls.append(recall)

    return np.mean(recalls)


def Recall_at_k_batch(X_pred, heldout_batch, k=10):
    """
    Compute Recall@k for binary relevance.

    Parameters:
    - X_pred: numpy.ndarray, predicted scores for all items
    - heldout_batch: numpy.ndarray or csr_matrix, true interactions for each user
    - k: int, cutoff for Recall@k

    Returns:
    - recall: numpy.ndarray, Recall@k for each user in the batch
    """
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    # Convert heldout_batch to dense array if it's a sparse matrix
    if isinstance(heldout_batch, np.ndarray):
        X_true_binary = heldout_batch > 0
    else:
        X_true_binary = (heldout_batch > 0).toarray()

    # Compute Recall@k
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    """
    Compute Normalized Discounted Cumulative Gain@k for binary relevance.

    Parameters:
    - X_pred: numpy.ndarray, predicted scores for all items
    - heldout_batch: numpy.ndarray or csr_matrix, true interactions for each user
    - k: int, cutoff for NDCG@k

    Returns:
    - ndcg: numpy.ndarray, NDCG@k for each user in the batch
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    # Convert heldout_batch to dense array if it's a sparse matrix
    if isinstance(heldout_batch, np.ndarray):
        X_true_binary = heldout_batch > 0
    else:
        X_true_binary = (heldout_batch > 0).toarray()

    # Compute DCG
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    DCG = (X_true_binary[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(
        axis=1
    )

    # Compute IDCG
    IDCG = np.array([tp[: min(n, k)].sum() for n in X_true_binary.sum(axis=1)])
    ndcg = DCG / IDCG
    ndcg[np.isnan(ndcg)] = 0.0  # Handle NaN for users with no interactions
    return ndcg
