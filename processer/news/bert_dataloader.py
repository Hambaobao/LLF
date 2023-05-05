#Coding: UTF-8
import math
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

from transformers import BertTokenizer as BertTokenizer
import utils.nlp_data_utils as data_utils
from utils.nlp_data_utils import ABSATokenizer

from datasets import load_dataset

classes_20newsgroup = [
    "19997_comp.graphics",
    "19997_comp.os.ms-windows.misc",
    "19997_comp.sys.ibm.pc.hardware",
    "19997_comp.sys.mac.hardware",
    "19997_comp.windows.x",
    "19997_rec.autos",
    "19997_rec.motorcycles",
    "19997_rec.sport.baseball",
    "19997_rec.sport.hockey",
    "19997_sci.crypt",
    "19997_sci.electronics",
    "19997_sci.med",
    "19997_sci.space",
    "19997_misc.forsale",
    "19997_talk.politics.misc",
    "19997_talk.politics.guns",
    "19997_talk.politics.mideast",
    "19997_talk.religion.misc",
    "19997_alt.atheism",
    "19997_soc.religion.christian",
]


def create_dataloader(data, batch_size):

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=True)

    return dataloader


def examples2data(config, examples, labels, tokenizer, t):
    features = data_utils.convert_examples_to_features_dtc(examples, labels, config['max_seq_length'], tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_tasks = torch.tensor([t for _ in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

    return data


def get(logger, config):
    """
    load data for dataset_name
    """
    dataset_name = 'newsgroup'
    classes = classes_20newsgroup

    print('dataset_name: ', dataset_name)

    data = {}

    # Others
    f_name = dataset_name + '_random_' + str(config.ntasks)
    with open("randseqs/" + f_name, 'r') as f_random_seq:
        random_seq = f_random_seq.readlines()[config.idrandom].split()

    logger.info('random sequence: {}'.format(str(random_seq)))

    dataset = {}
    dataset['train'] = {}
    dataset['valid'] = {}
    dataset['test'] = {}

    for c_id, cla in enumerate(classes):
        d = load_dataset(dataset_name, cla, split='train')
        d_split = d.train_test_split(test_size=0.2, shuffle=True, seed=config.seed)
        dataset['train'][c_id] = d_split['train']

        d_split = d_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=config.seed)  # test into half-half

        dataset['test'][c_id] = d_split['test']
        dataset['valid'][c_id] = d_split['train']

    class_per_task = config.num_labels

    examples = {}
    for s in ['train', 'test', 'valid']:
        examples[s] = {}
        for c_id, c_data in dataset[s].items():
            nn = (c_id // class_per_task)  #which task_id this class belongs to

            if nn not in examples[s]:
                examples[s][nn] = []
            for c_dat in c_data:
                text = c_dat['text']
                label = c_id % class_per_task
                examples[s][nn].append((text, label))

    train_dataloaders, valid_dataloaders, eval_dataloaders = [], [], []

    for t in range(config.ntasks):
        t_seq = int(random_seq[t].split('_')[-1])
        data[t] = {}
        data[t]['ncla'] = class_per_task
        data[t]['name'] = dataset_name + '_' + str(t_seq)

        processor = data_utils.DtcProcessor()
        label_list = processor.get_labels(config.ntasks)

        tokenizer = ABSATokenizer.from_pretrained(config.bert_model)
        train_examples = processor._create_examples(examples[s][t_seq], "train")
        num_train_steps = int(math.ceil(len(train_examples) / config.train_batch_size)) * config.train_epochs

        data[t]['train'] = examples2data(config, train_examples, label_list, tokenizer, t)
        data[t]['num_train_steps'] = num_train_steps

        valid_examples = processor._create_examples(examples[s][t_seq], "valid")
        data[t]['valid'] = examples2data(config, valid_examples, label_list, tokenizer, t)

        processor = data_utils.DtcProcessor()
        label_list = processor.get_labels(config.ntasks)
        tokenizer = BertTokenizer.from_pretrained(config.bert_model)

        eval_examples = processor._create_examples(examples[s][t_seq], "test")
        data[t]['test'] = examples2data(config, eval_examples, label_list, tokenizer, t)

        train_dataloader = create_dataloader(data[t]['train'], config.train_batch_size)
        train_dataloaders.append(train_dataloader)

        valid_dataloader = create_dataloader(data[t]['valid'], config.eval_batch_size)
        valid_dataloaders.append(valid_dataloader)

        eval_dataloader = create_dataloader(data[t]['test'], config.eval_batch_size)
        eval_dataloaders.append(eval_dataloader)

    return train_dataloaders, valid_dataloaders, eval_dataloaders