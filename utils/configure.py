import argparse
import wandb
import yaml


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default=False, action='store_true', help="whether to use wandb sweep")
    parser.add_argument("--count", type=int, default=0, help="count of sweep")

    args = parser.parse_args()

    return args


def wandb_init(config_path='config/config.yaml'):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def wandb_sweep(config_path='config/sweep_config.yaml'):
    with open(config_path, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep=sweep_config, project="llf")

    return sweep_id