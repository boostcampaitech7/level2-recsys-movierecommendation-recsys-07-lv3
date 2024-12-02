import recforest
import yaml
import argparse

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--model', '-m', '--m', type=str)
args = parser.parse_args()
selected_model = args.model

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

dataset_config = config['dataset_config']
model_config = config['model_config'][selected_model]
model_config['name'] = selected_model

manager = recforest.Manager(dataset_config, model_config)
manager.train()