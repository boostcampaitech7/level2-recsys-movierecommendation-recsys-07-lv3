import recforest
import yaml
import argparse

# 터미널로부터 인자를 받아옵니다.
parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--model', '-m', '--m', type=str)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
selected_model = args.model
is_test = args.test

# 설정 파일을 불러옵니다.
with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# 설정 파일로부터 데이터셋과 모델 설정을 불러옵니다.
dataset_config = config['dataset_config']
model_config = config['model_config'][selected_model]
model_config['name'] = selected_model

# 매니저를 초기화하고 학습 또는 테스트를 수행합니다.
manager = recforest.Manager(dataset_config, model_config)
if is_test:
    manager.test()
else:
    manager.train()