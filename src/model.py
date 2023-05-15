import torch
import torch.nn as nn

from transformers import BertModel, BertConfig


def activation(inputs):
    outputs = inputs

    return outputs


class Pooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.num_neurons, config.num_neurons)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LIF(nn.Module):

    def __init__(self, config):
        super(LIF, self).__init__()
        self.num_neurons = config.num_neurons
        self.max_seq_length = config.max_seq_length
        self.thresh = torch.Tensor(config.num_neurons).cuda()
        self.accumulation = None

    def forward(self, inputs):
        outputs = []
        inputs = activation(inputs)

        # word by word processing
        for i in range(self.max_seq_length):
            if self.accumulation is None:
                self.accumulation = inputs[:, i, :]
            else:
                self.accumulation += inputs[:, i, :]

            # spikes occur when accumulation great than thresh
            spikes = self.accumulation.gt(self.thresh)

            # treat accumulation as amplitude
            outputs_ = spikes * self.accumulation

            # reset to 0 after spike
            self.accumulation -= outputs_

            outputs.append(outputs_)

        outputs = torch.stack([o for o in outputs], dim=1)

        # limit amplitude of spikes
        outputs = activation(outputs)

        return outputs


class SNN(nn.Module):

    def __init__(self, config):
        super(SNN, self).__init__()
        self.linear1 = nn.Linear(config.num_neurons, config.num_neurons)
        self.lif1 = LIF(config)

        self.linear2 = nn.Linear(config.num_neurons, config.num_neurons)
        self.lif2 = LIF(config)

        self.linear3 = nn.Linear(config.num_neurons, config.num_neurons)
        self.lif3 = LIF(config)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.lif1(outputs)

        outputs = self.linear2(outputs)
        outputs = self.lif2(outputs)

        outputs = self.linear3(outputs)
        outputs = self.lif3(outputs)

        return outputs


class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()
        conf = BertConfig.from_pretrained(config.bert_model)
        self.bert = BertModel.from_pretrained(config.bert_model, config=conf).eval()
        self.snn = SNN(config)
        self.pooler = Pooler(config)
        self.classifier = nn.Linear(config.num_neurons, config.num_labels)

    def forward(self, input_ids, input_mask, segment_ids):

        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, output_hidden_states=True)

            hidden_states = bert_outputs[2]

            # sum of last three layer
            snn_inputs = torch.stack(hidden_states[-3:]).sum(0)

        snn_outputs = self.snn(snn_inputs)

        pooled_output = self.pooler(snn_outputs)

        logits = self.classifier(pooled_output)

        return logits, None