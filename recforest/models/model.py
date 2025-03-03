from abc import abstractmethod
import torch

from ..data import Dataset

class Model(torch.nn.Module):
    """추천 모델의 추상 클래스입니다.
    """

    def __init__(self, dataset: Dataset, config: dict) -> None:
        """Model 클래스를 초기화합니다.

        Parameters
        ----------
        dataset : Dataset
            학습에 사용할 데이터셋 객체입니다.
        config : dict
            Trainer와 모델에 대한 설정을 담은 딕셔너리입니다.
        """
        super(Model, self).__init__()
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def get_topk(self, k: int) -> torch.Tensor:
        """사용자별 상위 k개 아이템을 예측합니다.

        예측된 아이템은 학습 데이터에 포함된 아이템이어서는 안됩니다.

        Parameters
        ----------
        k : int
            예측할 아이템의 개수입니다.
        
        Returns
        -------
        torch.Tensor
            사용자별 상위 k개 아이템의 아이디입니다.
        """
        pass