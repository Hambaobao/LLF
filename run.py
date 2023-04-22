from processer.dsc import bert_dataloader as dataloader
from utils import configure, log, reproduce
from src.master import Master
from src.bert import Net

import wandb


def main(config=None):
    with wandb.init(config=config):

        logger.info('setting random seed to {}'.format(wandb.config.seed))
        reproduce.set_seed(wandb.config.seed)

        logger.info('start loading data ...')
        train_dataloaders, valid_dataloaders, eval_dataloaders = dataloader.get(logger=logger, config=wandb.config)
        logger.info('end loading data')

        logger.info('building model and put it on GPUs')
        net = Net(wandb.config).cuda()
        master = Master(wandb.config, logger, net)

        master.load_model(wandb.config.checkpoint_path)

        logger.info('start training ...')
        master.run(wandb, train_dataloaders, valid_dataloaders, eval_dataloaders)
        logger.info("end training")


if __name__ == "__main__":

    logger = log.set_logger()

    args = configure.parse()

    if args.sweep:
        sweep_id = configure.wandb_sweep()
        wandb.agent(sweep_id=sweep_id, function=main, count=args.count)
    else:
        config = configure.wandb_init()
        main(config)