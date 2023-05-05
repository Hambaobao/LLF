import math
import random

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

from transformers import BertTokenizer

import utils.nlp_data_utils as data_utils
from utils.nlp_data_utils import ABSATokenizer

domains = ['Video_Games', 'Toys_and_Games', 'Tools_and_Home_Improvement', 'Sports_and_Outdoors', 'Pet_Supplies', 'Patio_Lawn_and_Garden', 'Office_Products', 'Musical_Instruments', 'Movies_and_TV', 'Kindle_Store']

datasets = ['/data3/zl/lifelong/dat/dsc/' + domain for domain in domains]


def create_dataloader(data, batch_size):

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=True)

    return dataloader


def examples2data(config, examples, labels, tokenizer, t):
    features = data_utils.convert_examples_to_features_dsc(examples, labels, config['max_seq_length'], tokenizer, "dsc")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_tasks = torch.tensor([t for _ in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

    return data


def get(logger=None, config=None, f_name='randseqs/dsc_random_10'):
    data = {}

    with open(f_name, 'r') as f_random_seq:
        random_seq = f_random_seq.readlines()[config.idrandom].split()

    logger.info('random sequence: {}'.format(str(random_seq)))

    train_dataloaders, valid_dataloaders, eval_dataloaders = [], [], []

    for t in range(config.ntasks):
        dataset = datasets[domains.index(random_seq[t])]

        data[t] = {}
        data[t]['name'] = dataset

        logger.info('processing dataset {}: '.format(t + 1) + random_seq[t])

        processor = data_utils.DscProcessor()
        label_list = processor.get_labels()

        tokenizer = ABSATokenizer.from_pretrained(config.bert_model)
        train_examples = processor.get_train_examples(dataset)

        if config.train_data_size > 0:  #TODO: for replicated results, better do outside (in prep_dsc.py), so that can save as a file
            random.Random(config.data_seed).shuffle(train_examples)  #more robust
            train_examples = train_examples[:config.train_data_size]

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
