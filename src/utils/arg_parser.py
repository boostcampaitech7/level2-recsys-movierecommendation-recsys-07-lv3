import argparse
import ast
from omegaconf import OmegaConf
import torch.optim


def parse_args():
    """
    명령줄 인자를 파싱하여 학습할 모델 옵션을 설정합니다.

    이 함수는 다음과 같은 명령줄 옵션을 지원합니다:
    -lgb: LightGBM 모델 학습
    -cat: CatBoost 모델 학습
    -rf: Random Forest 모델 학습

    Returns:
        argparse.Namespace: 파싱된 명령줄 인자를 포함하는 Namespace 객체

    Example:
        python script.py -lgb -cat
        이 명령은 LightGBM과 CatBoost 모델을 학습하도록 설정합니다.
    """
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument

    # add basic arguments
    arg(
        "--config",
        "-c",
        "--c",
        type=str,
        help="Configuration 파일을 설정합니다.",
        default="configs/config.yaml",
    )
    arg(
        "--predict",
        "-p",
        "--p",
        "--pred",
        action="store_const",
        const=True,
        help="학습을 생략할지 여부를 설정할 수 있습니다.",
    )
    arg(
        "--checkpoint",
        "-ckpt",
        "--ckpt",
        type=str,
        help="학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.",
    )
    arg(
        "--model",
        "-m",
        "--m",
        type=str,
        choices=["MultiVAE", "EASE"],
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    arg(
        "--seed",
        "-s",
        "--s",
        type=int,
        help="데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.",
    )
    arg(
        "--device",
        "-d",
        "--d",
        type=str,
        choices=["cuda", "cpu", "mps"],
        help="사용할 디바이스를 선택할 수 있습니다.",
    )
    arg(
        "--wandb",
        "--w",
        "-w",
        action="store_const",
        const="True",
        help="wandb를 사용할지 여부를 설정할 수 있습니다.",
    )
    arg(
        "--wandb_project",
        "--wp",
        "-wp",
        type=str,
        help="wandb 프로젝트 이름을 설정할 수 있습니다.",
    )
    arg(
        "--run_name",
        "--rn",
        "-rn",
        "--r",
        "-r",
        type=str,
        help="wandb에서 사용할 run 이름을 설정할 수 있습니다.",
    )
    arg("--model_args", "--ma", "-ma", type=ast.literal_eval)
    arg("--dataloader", "--dl", "-dl", type=ast.literal_eval)
    arg("--dataset", "--dset", "-dset", type=ast.literal_eval)
    arg("--optimizer", "-opt", "--opt", type=ast.literal_eval)
    arg("--loss", "-l", "--l", type=str)
    arg("--lr_scheduler", "-lr", "--lr", type=ast.literal_eval)
    arg("--metrics", "-met", "--met", type=ast.literal_eval)
    arg("--train", "-t", "--t", type=ast.literal_eval)

    return parser.parse_args()


def parse_yaml(args):

    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    # 사용되지 않는 정보 삭제 (학습 시에만)
    if config_yaml.predict == False:
        del config_yaml.checkpoint

        if config_yaml.wandb == False:
            del config_yaml.wandb_project, config_yaml.run_name

        config_yaml.model_args = OmegaConf.create(
            {config_yaml.model: getattr(config_yaml, config_yaml.model)}
        )

        config_yaml.optimizer.args = {
            k: v
            for k, v in config_yaml.optimizer.args.items()
            if k
            in getattr(
                torch.optim, config_yaml.optimizer.type
            ).__init__.__code__.co_varnames
        }

        # if config_yaml.lr_scheduler.use == False:
        #     del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        # else:
        #     config_yaml.lr_scheduler.args = {
        #         k: v
        #         for k, v in config_yaml.lr_scheduler.args.items()
        #         if k
        #         in getattr(
        #             scheduler, config_yaml.lr_scheduler.type
        #         ).__init__.__code__.co_varnames
        #     }

        if config_yaml.train.resume == False:
            del config_yaml.train.resume_path

    return config_yaml
