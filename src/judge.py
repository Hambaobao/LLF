class Judger(object):

    def __init__(self):
        self.nsamples = 0
        self.accumulation = 0.
        self.accuracy = 0.

    def update(self, logits, targets):
        acc = self.calculate_accuracy(logits, targets)
        self.accumulation += acc
        self.nsamples += len(targets)
        self.accuracy = self.accumulation / self.nsamples

    def calculate_accuracy(self, logits, targets):
        _, predicts = logits.max(dim=1)
        hits = (predicts == targets).float()
        acc = hits.sum().data.cpu().numpy().item()

        return acc