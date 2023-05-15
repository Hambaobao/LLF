from processer.dsc import bert_dataloader as dsc_dataloader
from processer.asc import bert_dataloader as asc_dataloader
from processer.news import bert_dataloader as news_dataloader


def get_dataloader(args):
    if args.task == 'dsc':
        return dsc_dataloader
    elif args.task == 'asc':
        return asc_dataloader
    elif args.task == 'news':
        return news_dataloader
