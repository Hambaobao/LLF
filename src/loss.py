import torch.nn as nn


def calculate_loss(logits, targets):
    ce = nn.CrossEntropyLoss()
    loss_ce = ce(logits, targets)

    loss = loss_ce

    return loss