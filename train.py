import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from omegaconf import OmegaConf

from scipy.sparse import csr_matrix
from src.train.trainer import train, evaluate_MultiVAE
import src.train.loss as loss_module
import src.models as model_module
from src.data.preprocess import reindex_column, train_test_split
from src.utils.utils import set_seed
from src.utils.arg_parser import parse_args, parse_yaml
import src.data.dataset as dataset_module
from src.train.metric import recall_at_k_from_recommendations


def main(args):
    # seed 설정
    seed = args.seed
    set_seed(seed)

    ######################################## LOAD DATA ########################################

    data_path = args.dataset.data_path
    ratings = pd.read_csv(data_path + "train_ratings.csv")

    ######################################## PREPROCESS ########################################

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

    # data_path = args.dataset.data_path
    # data = getattr(dataset_module, args.dataset.data_type)(data_path)

    # train/valid 분할
    num_random_items = args.metrics.num_random_items  # random sampling 개수, 기본값 2개
    sequence = args.metrics.sequence  # sequnce sampling 여부, 기본값 True

    train_matrix, val_matrix = train_test_split(
        interaction_matrix, num_random_items, sequence
    )

    ######################################## LOGGING ########################################

    ######################################## MODEL ########################################

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print("device =", device)

    if args.model == "EASE":
        # model
        model_args = getattr(args, args.model)
        model = getattr(model_module, args.model)(**model_args)

        # no train, predict directly
        rating = model(train_matrix).cpu().numpy()  # np.ndarray (user x item)

        # Mask already interacted items
        masked_reconstructed = rating * (
            train_matrix.toarray() == 0
        )  # np.ndarray (user x item)

        # Get top-k recommendations for each user
        recommendations = []
        top_k = 10
        for user_id, scores in enumerate(masked_reconstructed):
            top_items = np.argsort(scores)[-top_k:][
                ::-1
            ]  # Get top-k items sorted by score
            for item_id in top_items:
                recommendations.append([user_id, item_id])

        for k in args.metrics.top_k:
            recall = recall_at_k_from_recommendations(recommendations, val_matrix, k)
            print(f"Recall@{k}: {recall:.4f}")

    if args.model == "MultiVAE":
        # Initialize model and loss function
        model_args = getattr(args, args.model)

        input_dim = train_matrix.shape[1]  # num_user
        model_args["p_dims"].append(
            input_dim
        )  # p_dims: [200, 600, input_dim] 같은 형식

        model = getattr(model_module, args.model)(**model_args)
        loss_fn = getattr(loss_module, args.loss)

        # Optimizer
        optim_args = (
            args.optimizer.args
        )  # ex) {'lr': 1e-3, 'weight_decay': 1e-4, 'armsgrad': False}
        # ex) optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, armsprad=False)
        optimizer = getattr(optim, args.optimizer.type)(
            model.parameters(), **optim_args
        )

    ######################################## TRAIN ########################################

    top_k = args.metrics.top_k  # ex) [10] or [5, 10]
    epochs = args.train.epochs

    # predict가 False 이면 학습/검증 단계까지만 진행
    if not args.predict:
        # EASE 모델은 closed form calculation 이므로 예외처리
        if not args.model == "EASE":
            for epoch in range(1, epochs + 1):
                print(f"Epoch {epoch}/{epochs}")
                train_loss = train(
                    model, loss_fn, optimizer, train_matrix, device, args
                )
                print(f"Training Loss: {train_loss:.4f}")

                val_loss, metrics = evaluate_MultiVAE(
                    model, train_matrix, val_matrix, args.metrics.top_k, loss_fn, args
                )
                print(f"Validation Loss: {val_loss:.4f}")
                for k in top_k:
                    print(
                        f"Recall@{k}: {metrics[f'Recall@{k}']:.4f}, NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}"
                    )
        # EASE 모델 계산
        # else:
        #     # no train, predict directly
        #     model.to(device)
        #     rating = model(train_matrix).cpu().numpy()

    ######################################## INFERENCE ########################################

    if args.predict:
        if args.model == "EASE":
            # no train, predict directly
            rating = model(interaction_matrix).cpu().numpy()

        ######################################## SAVE PREDICT ########################################

        # Mask already interacted items
        masked_reconstructed = rating * (interaction_matrix.toarray() == 0)

        # Get top-k recommendations for each user
        recommendations = []
        top_k = 10
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


if __name__ == "__main__":
    ######################################## CONFIG ########################################
    # cli args parsing
    args = parse_args()
    # config yaml pasing and overriding with cli args
    config_yaml = parse_yaml(args)

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))

    main(config_yaml)
