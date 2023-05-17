from utils.configure import set_args, random_name, wandb_sweep, wandb_init
from utils.reproduce import set_seed
from utils.log import set_logger
from utils.dataload import get_dataloader
from src.master import Master
from src.model import Net

import torch.nn as nn

import wandb

args = set_args()

logger = set_logger(args)

set_seed(args.seed)
logger.info('setting random seed to {}'.format(args.seed))

dataloader = get_dataloader(args)


def main(config=None):
    with wandb.init(reinit=True, config=config):

        if not wandb.run.name:
            wandb.run.name = random_name()

        logger.info('wandb run name {}'.format(wandb.run.name))

        logger.info('start loading data ...')
        train_dataloaders, valid_dataloaders, eval_dataloaders = dataloader.get(logger=logger, config=wandb.config)
        logger.info('end loading data')

        logger.info('building model and put it on GPUs')
        net = Net(wandb.config).cuda()
        # net = nn.DataParallel(net, device_ids=[0, 1], output_device=0)
        master = Master(wandb.config, logger, net)

        if args.resume:
            wandb.run.name = args.name
            master.load_model(wandb.config.checkpoint_path, wandb.run.name)

        logger.info('start training ...')
        master.run(wandb, train_dataloaders, valid_dataloaders, eval_dataloaders)
        logger.info("end training")

        wandb.run.name = None


if __name__ == "__main__":
    if args.sweep:
        sweep_id = wandb_sweep(args)
        wandb.agent(sweep_id=sweep_id, function=main, count=args.count)
    else:
        config = wandb_init(args)
        main(config)