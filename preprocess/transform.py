# Transform for caption
from torch.nn.utils.rnn import pad_sequence
import torch


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sentence):
        return self.tokenizer.tokenize(sentence)


class VocabToIndex:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, sentence):
        return [self.vocabulary.vocab2index(token) for token in sentence]


class ToLongTensor:
    def __call__(self, caption):
        return torch.LongTensor(caption)


class PadBatch:
    def __init__(self, padding_value=0):
        self.padding_value = padding_value

    def __call__(self, batch):
        # batch_tensor = [torch.LongTensor(data) for data in batch]
        return pad_sequence(batch, batch_first=True, padding_value=self.padding_value)
