#Coding: UTF-8
import math
import torch
from transformers import BertTokenizer as BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

import utils.nlp_data_utils as data_utils
from utils.nlp_data_utils import ABSATokenizer

datasets = [
    '/data3/zl/lifelong/dat/absa/XuSemEval/asc/14/rest',
    '/data3/zl/lifelong/dat/absa/XuSemEval/asc/14/laptop',
    '/data3/zl/lifelong/dat/absa/Bing3Domains/asc/Speaker',
    '/data3/zl/lifelong/dat/absa/Bing3Domains/asc/Router',
    '/data3/zl/lifelong/dat/absa/Bing3Domains/asc/Computer',
    '/data3/zl/lifelong/dat/absa/Bing5Domains/asc/Nokia6610',
    '/data3/zl/lifelong/dat/absa/Bing5Domains/asc/NikonCoolpix4300',
    '/data3/zl/lifelong/dat/absa/Bing5Domains/asc/CreativeLabsNomadJukeboxZenXtra40GB',
    '/data3/zl/lifelong/dat/absa/Bing5Domains/asc/CanonG3',
    '/data3/zl/lifelong/dat/absa/Bing5Domains/asc/ApexAD2600Progressive',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/CanonPowerShotSD500',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/CanonS100',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/DiaperChamp',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/HitachiRouter',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/ipod',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/LinksysRouter',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/MicroMP3',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/Nokia6600',
    '/data3/zl/lifelong/dat/absa/Bing9Domains/asc/Norton',
]

domains = [
    'XuSemEval14_rest', 'XuSemEval14_laptop', 'Bing3domains_Speaker', 'Bing3domains_Router', 'Bing3domains_Computer', 'Bing5domains_Nokia6610', 'Bing5domains_NikonCoolpix4300', 'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB', 'Bing5domains_CanonG3', 'Bing5domains_ApexAD2600Progressive', 'Bing9domains_CanonPowerShotSD500', 'Bing9domains_CanonS100', 'Bing9domains_DiaperChamp', 'Bing9domains_HitachiRouter', 'Bing9domains_ipod', 'Bing9domains_LinksysRouter', 'Bing9domains_MicroMP3',
    'Bing9domains_Nokia6600', 'Bing9domains_Norton'
]


def create_dataloader(data, batch_size):

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=True)

    return dataloader


def examples2data(config, examples, labels, tokenizer, t):
    train_features = data_utils.convert_examples_to_features(config, examples, labels, config['max_seq_length'], tokenizer, "dsc")

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_tasks = torch.tensor([t for _ in train_features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

    return data


def get(logger=None, config=None, f_name='./randoms/asc_random'):
    data = {}

    with open(f_name, 'r') as f_random_seq:
        random_seq = f_random_seq.readlines()[config.idrandom].split()

    logger.info('random sequence: {}'.format(str(random_seq)))

    train_dataloaders, valid_dataloaders, eval_dataloaders = [], [], []

    for t in range(config.ntasks):
        dataset = datasets[domains.index(random_seq[t])]

        data[t] = {}

        logger.info('processing dataset {}: '.format(t + 1) + random_seq[t])

        if 'Bing' in dataset:
            data[t]['ncla'] = 2
        elif 'XuSemEval' in dataset:
            data[t]['name'] = dataset
            data[t]['ncla'] = 3

        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(config.bert_model)
        train_examples = processor.get_train_examples(dataset)

        num_train_steps = int(math.ceil(len(train_examples) / config.train_batch_size)) * config.train_epochs

        data[t]['train'] = examples2data(config, train_examples, label_list, tokenizer, t)
        data[t]['num_train_steps'] = num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        data[t]['valid'] = examples2data(config, valid_examples, label_list, tokenizer, t)

        tokenizer = BertTokenizer.from_pretrained(config.bert_model)
        eval_examples = processor.get_test_examples(dataset)
        data[t]['test'] = examples2data(config, eval_examples, label_list, tokenizer, t)

        train_dataloader = create_dataloader(data[t]['train'], config.train_batch_size)
        train_dataloaders.append(train_dataloader)

        valid_dataloader = create_dataloader(data[t]['valid'], config.eval_batch_size)
        valid_dataloaders.append(valid_dataloader)

        eval_dataloader = create_dataloader(data[t]['test'], config.eval_batch_size)
        eval_dataloaders.append(eval_dataloader)

    return train_dataloaders, valid_dataloaders, eval_dataloaders