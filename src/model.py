import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertConfig


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
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length
        self.accumulations = torch.zeros(config.num_neurons)
        self.threshes = nn.Parameter(torch.rand(config.num_neurons))

    def get_goodness(self, inputs):
        # inputs: bath_size, max_seq_length, num_neurons, hidden_size
        # output: 1
        norm = torch.norm(inputs, dim=2) / inputs.shape[2]
        goodness = torch.sum(norm) / (inputs.shape[0] * inputs.shape[1] * inputs.shape[3])

        return goodness

    def activation(self, inputs):
        outputs = torch.sigmoid(inputs)

        return outputs

    def forward(self, inputs, data_type):
        inputs = inputs.detach()
        inputs.requires_grad = True

        outputs = []
        spikes = []

        inputs = self.activation(inputs)

        # word by word processing
        for i in range(self.max_seq_length):
            _outputs = []
            _spikes = []
            for j in range(self.hidden_size):
                _input = inputs[:, i, j]

                self.accumulations = self.accumulations.detach()
                self.accumulations.requires_grad = True

                accumulations = self.accumulations.to(_input.device) + _input.unsqueeze(dim=1)

                # spikes occur when accumulations great than thresh
                # batch_size, num_neurons
                spike = accumulations.gt(self.threshes)

                # treat accumulations as amplitude
                # batch_size, num_neurons
                output = spike * accumulations

                # reset to 0 after spike
                accumulations -= output

                self.accumulations = torch.max(accumulations, dim=0)[0]

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
        outputs = self.activation(outputs)

        goodness = self.get_goodness(outputs)

        if data_type == 'positive':
            loss = -goodness
        elif data_type == 'negative':
            loss = goodness
        else:
            print('data type error')

        return outputs, spikes, loss


class SNN(nn.Module):

    def __init__(self, config):
        super(SNN, self).__init__()
        self.linear1 = nn.Linear(config.bert_hidden_size, config.hidden_size)
        self.lif1 = LIF(config)
        self.integrator1 = Integrator(config)
        self.dropout1 = nn.Dropout(p=config.dropout_probability)
        self.params1 = list(self.linear1.parameters()) + list(self.lif1.parameters())
        self.optimizer1 = optim.SGD(self.params1, lr=config.learning_rate)

        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lif2 = LIF(config)
        self.integrator2 = Integrator(config)
        self.dropout2 = nn.Dropout(p=config.dropout_probability)
        self.params2 = list(self.linear1.parameters()) + list(self.lif1.parameters())
        self.optimizer2 = optim.SGD(self.params2, lr=config.learning_rate)

        self.linear3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lif3 = LIF(config)
        self.integrator3 = Integrator(config)
        self.dropout3 = nn.Dropout(p=config.dropout_probability)
        self.params3 = list(self.linear1.parameters()) + list(self.lif1.parameters())
        self.optimizer3 = optim.SGD(self.params3, lr=config.learning_rate)

        self.pooler = Pooler(config)

    def forward(self, inputs, data_type):
        hidden_spikes = []
        outputs = self.linear1(inputs)
        outputs, spikes, loss = self.lif1(outputs, data_type)
        outputs = self.integrator1(outputs)
        outputs = self.dropout1(outputs)
        hidden_spikes.append(spikes)
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()

        outputs = self.linear2(outputs)
        outputs, spikes, loss = self.lif2(outputs, data_type)
        outputs = self.integrator2(outputs)
        outputs = self.dropout2(outputs)
        hidden_spikes.append(spikes)
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer2.step()

        outputs = self.linear3(outputs)
        outputs, spikes, loss = self.lif3(outputs, data_type)
        outputs = self.integrator3(outputs)
        outputs = self.dropout3(outputs)
        hidden_spikes.append(spikes)
        self.optimizer3.zero_grad()
        loss.backward()
        self.optimizer3.step()

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

    def forward(self, input_ids, input_mask, segment_ids, data_type=None):

        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, output_hidden_states=True)

            hidden_states = bert_outputs[2]

            # sum of last three layer
            snn_inputs = torch.stack(hidden_states[-3:]).sum(0)

        snn_outputs, hidden_spikes = self.snn(snn_inputs, data_type='positive')

        logits = self.classifier(snn_outputs)

        return logits, hidden_spikes