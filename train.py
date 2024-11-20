import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import tqdm
from omegaconf import OmegaConf

from scipy.sparse import csr_matrix
from src.train.trainer import train, evaluate_MultiVAE
from src.train.loss import loss_function_vae
from src.models.MultiVAE import MultiVAE
from src.models.EASE import EASE
from src.data.preprocess import reindex_column, train_test_split
from src.utils.utils import set_seed


def main(args):
    # seed 설정
    seed = 42
    set_seed(seed)

    #################### LOAD DATA ####################
    data_path = "../../data/train/"  # 알잘딱 수정
    ratings = pd.read_csv(data_path + "train_ratings.csv")

    #################### PREPROCESS ####################
    ratings, usr2idx_dict = reindex_column(ratings, "user")
    ratings, item2idx_dict = reindex_column(ratings, "item")

    # csr 구성하는 row column define
    rows, cols = ratings["user"], ratings["item"]
    feedback = np.ones_like(rows)

    # csr matrix 생성
    num_users, num_items = ratings["user"].nunique(), ratings["item"].nunique()
    interaction_matrix = csr_matrix(
        (feedback, (rows, cols)), dtype="float64", shape=(num_users, num_items)
    )

    # train/valid 분할
    train_matrix, val_matrix = train_test_split(interaction_matrix)

    #################### MODEL ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_k = 10
    if args.model == "EASE":
        # model = EASE(train_matrix, device).to(device)
        # Test on the train + validation set
        adj_mat = convert_sp_mat_to_sp_tensor(interaction_matrix).to_dense()
        train_model = EASE(adj_mat, device)
        rating = train_model(1000).cpu().numpy()
        # Mask already interacted items
        masked_reconstructed = rating * (interaction_matrix.toarray() == 0)
        # Get top-k recommendations for each user
        recommendations = []
        for user_id, scores in enumerate(masked_reconstructed):
            top_items = np.argsort(scores)[-top_k:][
                ::-1
            ]  # Get top-k items sorted by score
            for item_id in top_items:
                recommendations.append([user_id, item_id])
        # Map back to original user and item IDs
        reverse_user_mapping = {v: k for k, v in usr2idx_dict.items()}
        reverse_item_mapping = {v: k for k, v in item2idx_dict.items()}
        recommendations_df = pd.DataFrame(recommendations, columns=["user", "item"])
        recommendations_df["user"] = recommendations_df["user"].map(
            reverse_user_mapping
        )
        recommendations_df["item"] = recommendations_df["item"].map(
            reverse_item_mapping
        )
        recommendations_df.to_csv("submission.csv", index=False)

    elif args.model == "MultiVAE":
        N = train_matrix.shape[0]

        # Hyperparameters
        learning_rate = 1e-3

        # Initialize model, optimizer, and loss function
        input_dim = train_matrix.shape[1]
        p_dims = [200, 600, input_dim]
        model = MultiVAE(p_dims).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=args.wd
        )

    #################### TRAIN ####################

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train(
            model, loss_function_vae, optimizer, train_matrix, device, args
        )
        print(f"Training Loss: {train_loss:.4f}")

        val_loss, metrics = evaluate_MultiVAE(
            model, train_matrix, val_matrix, loss_function_vae, device, top_k=[10]
        )
        print(f"Validation Loss: {val_loss:.4f}")
        for k in [10]:
            print(
                f"Recall@{k}: {metrics[f'Recall@{k}']:.4f}, NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}"
            )

    #################### INFERENCE ####################

    #################### SAVE PREDICT ####################


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


if __name__ == "__main__":
    #################### CONFIG ####################
    # arg parsing
    parser = argparse.ArgumentParser(
        description="PyTorch Variational Autoencoders for Collaborative Filtering"
    )
    arg = parser.add_argument

    arg("--lr", type=float, default=1e-4, help="initial learning rate")
    arg("--model", type=str, default="EASE", help="model name")
    arg("--wd", type=float, default=0.00, help="weight decay coefficient")
    arg("--batch_size", type=int, default=500, help="batch size")
    arg("--epochs", type=int, default=1, help="upper epoch limit")
    arg(
        "--total_anneal_steps",
        type=int,
        default=200000,
        help="the total number of gradient updates for annealing",
    )
    arg("--anneal_cap", type=float, default=0.2, help="largest annealing parameter")
    arg("--cuda", action="store_true", help="use CUDA")
    arg("--log_interval", type=int, default=100, metavar="N", help="report interval")
    arg("--save", type=str, default="model.pt", help="path to save the final model")

    args = parser.parse_args()

    # config_args = OmegaConf.create(vars(args))
    # config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    # for key in config_args.keys():
    #     if config_args[key] is not None:
    #         config_yaml[key] = config_args[key]

    # # Configuration 콘솔에 출력
    # print(OmegaConf.to_yaml(config_yaml))

    main(args)
