import torch
import torch.nn as nn

from transformers import BertModel, BertConfig


def activation(inputs):
    outputs = inputs

    return outputs


class Pooler(nn.Module):

    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        # inputs shape: bath_size, max_seq_length, hidden_size
        # outputs shape: bath_size, max_seq_length
        pooled_output = self.dense(inputs).squeeze(2)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class Integrator(nn.Module):

    def __init__(self, config):
        super(Integrator, self).__init__()
        self.num_neurons = config.num_neurons
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length

    def forward(self, inputs):
        # inputs shape: bath_size, max_seq_length, num_neurons, hidden_size
        # outputs shape: bath_size, max_seq_length, hidden_size
        outputs = torch.sum(inputs, dim=2)

        return outputs


class LIF(nn.Module):

    def __init__(self, config):
        super(LIF, self).__init__()
        self.num_neurons = config.num_neurons
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length
        self.threshes = torch.rand(config.num_neurons).cuda()
        self.accumulations = torch.zeros(config.train_batch_size, config.num_neurons).cuda()

        self.dropout = nn.Dropout(p=config.dropout_probability)

    def forward(self, inputs):
        outputs = []
        spikes = []
        inputs = activation(inputs)

        # word by word processing
        for i in range(self.max_seq_length):
            _outputs = []
            _spikes = []
            for j in range(self.hidden_size):
                self.accumulations += inputs[:, i, j].unsqueeze(dim=1)

                # do dropout
                self.accumulations = self.dropout(self.accumulations)

                # spikes occur when accumulations great than thresh
                # batch_size, num_neurons
                spike = self.accumulations.gt(self.threshes)

                # treat accumulations as amplitude
                # batch_size, num_neurons
                output = spike * self.accumulations

                # reset to 0 after spike
                self.accumulations -= output

                _outputs.append(output)
                _spikes.append(spike)

            # bath_size, num_neurons, hidden_size
            _outputs = torch.stack([o for o in _outputs], dim=2)
            _spikes = torch.stack([s for s in _spikes], dim=2)

            outputs.append(_outputs)
            spikes.append(_spikes)

        # bath_size, max_seq_length, num_neurons, hidden_size
        outputs = torch.stack([o for o in outputs], dim=1)
        spikes = torch.stack([s for s in spikes], dim=1)

        # limit amplitude of outputs
        outputs = activation(outputs)

        return outputs, spikes


class SNN(nn.Module):

    def __init__(self, config):
        super(SNN, self).__init__()
        self.linear1 = nn.Linear(config.bert_hidden_size, config.hidden_size)
        self.lif1 = LIF(config)
        self.integrator1 = Integrator(config)

        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lif2 = LIF(config)
        self.integrator2 = Integrator(config)

        self.linear3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lif3 = LIF(config)
        self.integrator3 = Integrator(config)

        self.pooler = Pooler(config)

    def forward(self, inputs):
        hidden_spikes = []
        outputs = self.linear1(inputs)
        outputs, spikes = self.lif1(outputs)
        outputs = self.integrator1(outputs)
        hidden_spikes.append(spikes)

        outputs = self.linear2(outputs)
        outputs, spikes = self.lif2(outputs)
        outputs = self.integrator2(outputs)
        hidden_spikes.append(spikes)

        outputs = self.linear3(outputs)
        outputs, spikes = self.lif3(outputs)
        outputs = self.integrator3(outputs)
        hidden_spikes.append(spikes)

        outputs = self.pooler(outputs)

        hidden_spikes = torch.stack([h for h in hidden_spikes], dim=0)

        return outputs, hidden_spikes


class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()
        conf = BertConfig.from_pretrained(config.bert_model)
        self.bert = BertModel.from_pretrained(config.bert_model, config=conf).eval()
        self.snn = SNN(config)
        self.classifier = nn.Linear(config.max_seq_length, config.num_labels)

    def forward(self, input_ids, input_mask, segment_ids):

        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, output_hidden_states=True)

            hidden_states = bert_outputs[2]

            # sum of last three layer
            snn_inputs = torch.stack(hidden_states[-3:]).sum(0)

            snn_outputs, hidden_spikes = self.snn(snn_inputs)

            logits = self.classifier(snn_outputs)

        return logits, hidden_spikes