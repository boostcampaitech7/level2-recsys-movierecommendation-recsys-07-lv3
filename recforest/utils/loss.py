import torch

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

def ccl_loss(pos_scores: torch.Tensor, neg_scores_list: list[torch.Tensor]) -> torch.Tensor:
    w, m = 0.5, 300
    pos_loss = 1 - pos_scores
    neg_loss = 0
    for neg_scores in neg_scores_list:
        neg_loss += torch.maximum(torch.zeros_like(neg_scores), neg_scores - m)
    neg_loss /= len(neg_scores_list)
    loss = pos_loss + w * neg_loss
    return torch.mean(loss)

def rmse_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((true - pred) ** 2))

def vae_bce_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return -torch.sum(torch.nn.functional.log_softmax(pred, 1) * true, -1)

def vae_reg_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1)