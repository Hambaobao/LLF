import torch
import torch.optim as optim

import json
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable

from src.judge import Judger
from src.loss import calculate_loss


def create_optimizer(wandb, model):
    optimizer = optim.Adam(model.parameters(), lr=float(wandb.config['learning_rate']))

    return optimizer


def create_scheduler(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=1e-4)

    return scheduler


class Recorder(object):

    def __init__(self, config):
        self.temp_row = ['' for _ in range(config.ntasks)]

        head = ['task {}'.format(i) for i in range(config.ntasks)]
        self.records = PrettyTable(head)
        self.average_accuracy = 0.

    def add_record(self, record):
        re = ['{:.2f}%'.format(100 * r) for r in record]
        self.records.add_row(re)
        self.average_accuracy = sum(record) / len(record)

    def print_records(self):
        print(self.records)


class Master(object):

    def __init__(self, config, logger, model):
        self.logger = logger
        self.model = model
        self.valid_recorder = Recorder(config)
        self.eval_recorder = Recorder(config)

        self.init_t = 0
        self.init_e = 0

    def run(self, wandb, train_dataloaders, valid_dataloaders, eval_dataloaders):
        for t in range(self.init_t, wandb.config.ntasks):
            self.logger.info('training on dataset {} ...'.format(t))

            self.train(wandb, train_dataloaders, t)

            self.logger.info('validing after trained on dataset 0-{} ...'.format(t))

            self.valid(wandb, valid_dataloaders, t)

            self.logger.info('evaluating after trained on dataset 0-{} ...'.format(t))

            self.evaluate(wandb, eval_dataloaders, t)

        return self.eval_recorder.average_accuracy

    def train(self, wandb, train_dataloaders, t):
        self.model.train()

        optimizer = create_optimizer(wandb, self.model)
        scheduler = create_scheduler(optimizer)

        train_dataloader = train_dataloaders[t]
        for e in range(self.init_e, wandb.config.train_epochs):
            trainer = Judger()
            with tqdm(total=len(train_dataloader), disable=False, ascii=True) as p:
                p.set_description("train epoch {}".format(e))
                for data in train_dataloader:
                    data = [d.cuda() for d in data]
                    input_ids, segment_ids, input_mask, targets, _ = data

                    outputs = self.model(input_ids, input_mask, segment_ids)

                    logits = outputs[0]

                    loss = calculate_loss(logits, targets)

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()
                    scheduler.step()

                    trainer.update(logits, targets)

                    p.set_postfix({'loss': "{:.2f}".format(loss.item()), 'accuracy': '{:.1f}%'.format(100 * trainer.accuracy)})
                    p.update(1)

            wandb.log({"train accuracy": trainer.accuracy, "loss": loss})
            self.logger.info('saving model checkpoint of task: {}, epoch: {}'.format(t, e))
            self.save_model(wandb, t, e)

        self.init_e = 0

    def valid(self, wandb, valid_dataloaders, t):
        self.model.eval()

        record = [0. for _ in range(wandb.config.ntasks)]
        for i in range(t + 1):
            valider = Judger()
            valid_dataloader = valid_dataloaders[i]
            with tqdm(total=len(valid_dataloader), disable=False, ascii=True) as p:
                p.set_description("dataset {}".format(i + 1))
                for data in valid_dataloader:
                    data = [d.cuda() for d in data]
                    input_ids, segment_ids, input_mask, targets, _ = data

                    outputs = self.model(input_ids, input_mask, segment_ids)

                    logits = outputs[0]

                    valider.update(logits, targets)

                    loss = calculate_loss(logits, targets)

                    p.set_postfix({'loss': "{:.2f}".format(loss.item()), 'accuracy': '{:.1f}%'.format(100 * valider.accuracy)})
                    p.update(1)

            record[i] = valider.accuracy

        self.valid_recorder.add_record(record)
        self.logger.info('\n' + self.valid_recorder.records.get_string())

    def evaluate(self, wandb, eval_dataloaders, t):
        self.model.eval()

        record = [0. for _ in range(wandb.config.ntasks)]

        for i in range(t + 1):
            evaluator = Judger()
            eval_dataloader = eval_dataloaders[i]
            with tqdm(total=len(eval_dataloader), disable=False, ascii=True) as p:
                p.set_description("dataset {}".format(i))
                for data in eval_dataloader:
                    data = [d.cuda() for d in data]
                    input_ids, segment_ids, input_mask, targets, _ = data

                    outputs = self.model(input_ids, input_mask, segment_ids)

                    logits = outputs[0]

                    evaluator.update(logits, targets)

                    loss = calculate_loss(logits, targets)

                    p.set_postfix({'loss': "{:.2f}".format(loss.item()), 'accuracy': '{:.1f}%'.format(100 * evaluator.accuracy)})
                    p.update(1)

            record[i] = evaluator.accuracy

        self.eval_recorder.add_record(record)
        self.logger.info('\n' + self.eval_recorder.records.get_string())

    def save_model(self, wandb, t, e):
        checkpoint_path = Path(wandb.config.checkpoint_path)

        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        info = {'task': t, 'epoch': e}
        with open(checkpoint_path / "info.json", "w") as f:
            json.dump(info, f)

        checkpoint_path /= Path('task_{0}_epoch_{1}.pt'.format(t, e))

        torch.save(self.model.state_dict(), checkpoint_path)

    def load_model(self, path):
        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            self.init_t, self.init_e = self.check_history(checkpoint_path)

            checkpoint_path /= Path('task_{0}_epoch_{1}.pt'.format(self.init_t, self.init_e))
            self.model.load_state_dict(torch.load(checkpoint_path))

            # recovered trainning process will start from e + 1 epoch
            self.init_e += 1
            self.logger.info('recovered trainning process will start from task {}, epoch {}'.format(self.init_t, self.init_e))

    def check_history(self, checkpoint_path):
        with open(checkpoint_path / 'info.json') as f:
            info = json.load(f)
            t = info['task']
            e = info['epoch']

        return t, e
