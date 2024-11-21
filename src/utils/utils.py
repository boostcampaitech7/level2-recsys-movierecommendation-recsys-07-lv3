import os
import time
import random
import logging
import numpy as np
import torch
from omegaconf import OmegaConf


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# class Setting():
#     def __init__(self):
#         now = time.localtime()
#         now_date = time.strftime("%Y%m%d", now)
#         now_hour = time.strftime("%X", now)
#         save_time = now_date + "_" + now_hour.replace(":", "")
#         self.save_time = save_time

#     def set_seed(seed):
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed) # if use multi-GPU
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         np.random.seed(seed)
#         random.seed(seed)

#     def set_wandb():
#         # TODO
#         pass

#     def get_log_path(self, args):
#         """
#         [description]
#         log file을 저장할 경로를 반환하는 함수입니다.

#         [arguments]
#         args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

#         [return]
#         path : log file을 저장할 경로를 반환합니다.
#         이 때, 경로는 saved/log/날짜_시간_모델명/ 입니다.
#         """
#         path = os.path.join(args.train.log_dir, f"{self.save_time}_{args.model}/")
#         self.make_dir(path)

#         return path

#     def get_submit_filename(self, args):
#         """
#         [description]
#         submit file을 저장할 경로를 반환하는 함수입니다.

#         [arguments]
#         args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

#         [return]
#         filename : submit file을 저장할 경로를 반환합니다.
#         이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
#         """
#         if args.predict == False:
#             self.make_dir(args.train.submit_dir)
#             filename = os.path.join(
#                 args.train.submit_dir, f"{self.save_time}_{args.model}.csv"
#             )
#         else:
#             filename = os.path.basename(args.checkpoint)
#             filename = os.path.join(args.train.submit_dir, f"{filename}.csv")

#         return filename

#     def make_dir(self, path):
#         """
#         [description]
#         경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

#         [arguments]
#         path : 경로

#         [return]
#         path : 경로
#         """
#         if not os.path.exists(path):
#             os.makedirs(path)
#         else:
#             pass
#         return path
#     pass


# class Logger():
#     def __init__(self, args, path):
#         """
#         [description]
#         log file을 생성하는 클래스입니다.

#         [arguments]
#         args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
#         path : log file을 저장할 경로를 전달받습니다.
#         """
#         self.args = args
#         self.path = path

#         self.logger = logging.getLogger()
#         self.logger.setLevel(logging.INFO)
#         self.formatter = logging.Formatter("[%(asctime)s] - %(message)s")

#         self.file_handler = logging.FileHandler(os.path.join(self.path, "train.log"))
#         self.file_handler.setFormatter(self.formatter)
#         self.logger.addHandler(self.file_handler)

#     def log(self, epoch, train_loss, valid_loss=None, valid_metrics=None):
#         """
#         [description]
#         log file에 epoch, train loss, valid loss를 기록하는 함수입니다.
#         이 때, log file은 train.log로 저장됩니다.

#         [arguments]
#         epoch : epoch
#         train_loss : train loss
#         valid_loss : valid loss
#         """
#         message = (
#             f"epoch : {epoch}/{self.args.train.epochs} | train loss : {train_loss:.3f}"
#         )
#         if valid_loss:
#             message += f" | valid loss : {valid_loss:.3f}"
#         if valid_metrics:
#             for metric, value in valid_metrics.items():
#                 message += f" | valid {metric.lower()} : {value:.3f}"
#         self.logger.info(message)

#     def close(self):
#         """
#         [description]
#         log file을 닫는 함수입니다.
#         """
#         self.logger.removeHandler(self.file_handler)
#         self.file_handler.close()

#     def save_args(self):
#         """
#         [description]
#         model에 사용된 args를 저장하는 함수입니다.
#         이 때, 저장되는 파일명은 model.json으로 저장됩니다.
#         """
#         with open(os.path.join(self.path, "config.yaml"), "w") as f:
#             OmegaConf.save(self.args, f)

#     def __del__(self):
#         self.close()


# def convert_sp_mat_to_sp_tensor(X: csr_matrix) -> torch.sparse.FloatTensor:
#     coo = X.tocoo().astype(np.float32)
#     row = torch.Tensor(coo.row).long()
#     col = torch.Tensor(coo.col).long()
#     index = torch.stack([row, col])
#     data = torch.FloatTensor(coo.data)
#     return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


# def sparse2torch_sparse(data):
#     """
#     Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
#     This is much faster than naive use of torch.FloatTensor(data.toarray())
#     https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
#     """
#     row = data.shape[0]
#     col = data.shape[1]
#     coo_data = data.tocoo()
#     indices = torch.LongTensor([coo_data.row, coo_data.col])
#     row_norms_inv = 1 / np.sqrt(data.sum(1))
#     row2val = {i : row_norms_inv[i].item() for i in range(row)}
#     values = np.array([row2val[r] for r in coo_data.row])
#     t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [row, col])
#     return t


def sparse2torch_sparse(data):
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
