import argparse
import json

def retrieve_best_ckpt(dir):
    path = f'{dir}/trainer_state.json'
    with open(path) as f:
        line = json.load(f)
        name = line['best_model_checkpoint']
        name = name.split('/')[-1]
    return dir + '/' + name
