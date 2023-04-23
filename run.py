from processer.dsc import bert_dataloader as dataloader
from utils import configure, log, reproduce
from src.master import Master
from src.bert import Net

import wandb


def main(config=None):
    with wandb.init(reinit=True, config=config):

        if not wandb.run.name:
            wandb.run.name = configure.random_name()

        logger.info('wandb run name {}'.format(wandb.run.name))

        logger.info('start loading data ...')
        train_dataloaders, valid_dataloaders, eval_dataloaders = dataloader.get(logger=logger, config=wandb.config)
        logger.info('end loading data')

        logger.info('building model and put it on GPUs')
        net = Net(wandb.config).cuda()
        master = Master(wandb.config, logger, net)

        if args.resume:
            wandb.run.name = args.name
            master.load_model(wandb.config.checkpoint_path, wandb.run.name)

        logger.info('start training ...')
        master.run(wandb, train_dataloaders, valid_dataloaders, eval_dataloaders)
        logger.info("end training")

        wandb.run.name = None


if __name__ == "__main__":

    logger = log.set_logger()

    args = configure.parse()

    logger.info('setting random seed to {}'.format(args.seed))
    reproduce.set_seed(args.seed)

    if args.sweep:
        sweep_id = configure.wandb_sweep(args)
        wandb.agent(sweep_id=sweep_id, function=main, count=args.count)
    else:
        config = configure.wandb_init(args)
        main(config)