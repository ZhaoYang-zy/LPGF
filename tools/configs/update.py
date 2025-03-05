import json
import os


def update_configs(args, exp_name, model):
    path = os.path.dirname(__file__)
    path = os.path.join(path, exp_name, f'{model}.json')
  
  
    with open(path, 'r') as f:
        new = json.load(f)

    conf =  args.__dict__
    conf.update(new)