from utils import configure, log, reproduce
from src.master import Master
from src.bert import Net

import wandb

logger = log.set_logger()

args = configure.set_args()

reproduce.set_seed(args.seed)
logger.info('setting random seed to {}'.format(args.seed))

if args.task == 'dsc':
    from processer.dsc import bert_dataloader as dataloader
elif args.task == 'asc':
    from processer.asc import bert_dataloader as dataloader
elif args.task == 'news':
    from processer.news import bert_dataloader as dataloader


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
    if args.sweep:
        sweep_id = configure.wandb_sweep(args)
        wandb.agent(sweep_id=sweep_id, function=main, count=args.count)
    else:
        config = configure.wandb_init(args)
        main(config)