# 🎞️ Movie Recommendation
> 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 테스크입니다.

## Team
|곽정무|박준하|박태지|배현우|신경호|이효준
|:-:|:-:|:-:|:-:|:-:|:-:
|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/20788198?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/81938013?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/112858891?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/179800298?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/103016689?s=64&v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/176903280?v=4'>|
|<a href = 'https://github.com/jkwag'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/joshua5301'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/spsp4755'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/hwbae42'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/Human3321'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/Jun9096'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|






## 프로젝트 구조
```
📦level2-recsys-movierecommendation-recsys-07-lv3
 ┣ 📂recforest
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜encoder.py
 ┃ ┃ ┣ 📜loader.py
 ┃ ┃ ┣ 📜splitter.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📂EASE
 ┃ ┃ ┃ ┣ 📜EASE.py
 ┃ ┃ ┃ ┣ 📜EASE_trainer.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂LightGCN
 ┃ ┃ ┃ ┣ 📜LightGCN.py
 ┃ ┃ ┃ ┣ 📜LightGCN_trainer.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂MultiVAE
 ┃ ┃ ┃ ┣ 📜MultiVAE.py
 ┃ ┃ ┃ ┣ 📜MultiVAE_trainer.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂RecVAE
 ┃ ┃ ┃ ┣ 📜RecVAE.py
 ┃ ┃ ┃ ┣ 📜RecVAE_trainer.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂SASRec
 ┃ ┃ ┃ ┣ 📜modules.py
 ┃ ┃ ┃ ┣ 📜SASRec.py
 ┃ ┃ ┃ ┣ 📜SASRec_trainer.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂SVDAE
 ┃ ┃ ┃ ┣ 📜SVDAE.py
 ┃ ┃ ┃ ┣ 📜SVDAE_trainer.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┣ 📜trainer.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📜loss.py
 ┃ ┃ ┣ 📜metric.py
 ┃ ┃ ┗ 📜sampler.py
 ┃ ┣ 📜manager.py
 ┃ ┗ 📜__init__.py
 ┣ 📂sequential
 ┃ ┣ 📜datasets.py
 ┃ ┣ 📜inference.py
 ┃ ┣ 📜models.py
 ┃ ┣ 📜modules.py
 ┃ ┣ 📜run_train.py
 ┃ ┣ 📜trainers.py
 ┃ ┗ 📜utils.py
 ┣ 📜config.yaml
 ┣ 📜main.py
 ┗ 📜README.md
```

## 개발환경
- python 3.10.0

## 기술스택
<img src = 'https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54'> <img src = 'https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white'> <img src= 'https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white'> <img src ='https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white'> 

### 협업툴
<img src ='https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white'> <img src = 'https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white'>


## 라이브러리 설치
```shell
$ pip install -r requirement.txt
```

## 기능 및 예시
- Static model

EASE, RecVAE 등 static model을 실행하는 script입니다.

```shell
$ python main.py -m {model명}
```

제출을 위한 테스트를 진행하려면 --test 인자를 추가해주세요.

```shell
$ python main.py -m {model명} --test
```

<br/>

- Sequential model (SASRec)

Sequential 데이터를 예측하는 SASRec을 학습시키는 코드입니다.

파일 경로를 sequential로 설정한 후 아래 script를 실행합니다.
```shell
$ python run_train.py
```
그 후 아래의 script를 실행하면 top-10 recommandation을 추천합니다.
```shell
$ python inference.py
```


