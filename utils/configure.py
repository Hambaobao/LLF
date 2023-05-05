import argparse
import wandb
import yaml
import random
import string


def random_name(k=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))


def parse(parser):
    parser.add_argument("--task", required=True, type=str, choices=['dsc', 'asc', 'news'], help="tasks")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")

    parser.add_argument("--resume", default=False, action='store_true', help="whether resume from last train")
    parser.add_argument("--name", type=str, default='', help="name of resume train")

    parser.add_argument("--sweep", default=False, action='store_true', help="whether to use wandb sweep")
    parser.add_argument("--count", type=int, default=0, help="count of sweep")

    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = parse(parser)
    args = parser.parse_args()

    return args


def wandb_init(args):
    config_path = 'config/' + args.task + '/config.yaml'
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def wandb_sweep(args):
    config_path = 'config/' + args.task + '/sweep_config.yaml'
    with open(config_path, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep=sweep_config, project="llf")

    return sweep_id