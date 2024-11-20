import numpy as np
import pandas as pd
import warnings
import argparse
import tqdm
import wandb
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


from src.utils.utils import Setting, Logger
import src.utils as utils
from src.utils.arg_parser import parse_args, parse_yaml
import src.data as data_module
import src.models as model_module
from src.train.trainer import train, test

# warnings.filterwarnings('ignore')


def main(cfg):
    setting = Setting()
    # wandb 사용 설정 시 wandb init
    if args.wandb:
        setting.set_wandb(cfg.wandb)
    # 시드 설정
    setting.set_seed(cfg.seed)

    ################ LOAD DATA
    datatype = args.model_args[args.model].datatype
    data_load_fn = getattr(
        data_module, f"{datatype}_data_load"
    )  # e.g. basic_data_load()
    data_split_fn = getattr(
        data_module, f"{datatype}_data_split"
    )  # e.g. basic_data_split()
    data_loader_fn = getattr(
        data_module, f"{datatype}_data_loader"
    )  # e.g. basic_data_loader()

    print(f"--------------- {args.model} Load Data ---------------")
    data = data_load_fn(args)

    print(f"--------------- {args.model} Train/Valid Split ---------------")
    data = data_split_fn(args, data)
    data = data_loader_fn(args, data)

    ################ LOGGING
    if args.predict == False:
        log_path = setting.get_log_path(args)
        logger = Logger(args, log_path)
        logger.save_args()

    ################ MODEL
    print(f"--------------- INIT {args.model} ---------------")
    # models > __init__.py 에 저장된 모델만 사용 가능
    # model = FM(args.model_args.FM, data).to('cuda')와 동일한 코드
    model = getattr(model_module, args.model)(args.model_args[args.model], data).to(
        args.device
    )

    # 만일 기존의 모델을 불러와서 학습을 시작하려면 resume을 true로 설정하고 resume_path에 모델을 지정하면 됨
    if args.train.resume:
        model.load_state_dict(torch.load(args.train.resume_path, weights_only=True))

    ################ TRAIN
    if not args.predict:
        print(f"--------------- {args.model} TRAINING ---------------")
        model = train(args, model, data, logger, setting)

    ################ INFERENCE
    if not args.predict:
        print(f"--------------- {args.model} PREDICT ---------------")
        predicts = test(args, model, data, setting)
    else:
        print(f"--------------- {args.model} PREDICT ---------------")
        predicts = test(args, model, data, setting, args.checkpoint)

    ################ SAVE PREDICT
    print(f"--------------- SAVE {args.model} PREDICT ---------------")
    submission = pd.read_csv(args.dataset.data_path + "sample_submission.csv")
    submission["rating"] = predicts

    filename = setting.get_submit_filename(args)
    print(f"Save Predict: {filename}")
    submission.to_csv(filename, index=False)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    ################ BASIC ENV SETUP
    args = parse_args()

    ################ yaml CONFIG
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    # 사용되지 않는 정보 삭제 (학습 시에만)
    parse_yaml(config_args, config_yaml, optim, scheduler)

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))

    ################ MAIN
    main(config_yaml)
