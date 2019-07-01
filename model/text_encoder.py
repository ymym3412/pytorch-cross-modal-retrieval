from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
import torch
import numpy as np
from .model_util import l2norm


class TextEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, dropout, num_layers, use_abs=False):
        super(TextEncoder, self).__init__()
        self.use_abs = use_abs
        self.hidden_size = hidden_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # caption embedding
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        # Linear
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        # pytorch 1.1
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded, _ = pad_packed_sequence(out, batch_first=True)

        # Use final LSTM Output
        length_tensor = lengths.reshape(-1, 1, 1)  # shape: bs,1,1
        # Get last index (before padding) of each sentence
        length_tensor = length_tensor.expand(x.size(0), 1, self.hidden_size) - 1  # bs, 1, hidden_size
        out = torch.gather(padded, 1, length_tensor).squeeze(1)  # bs * hidden_size

        # normalization in the joint embedding space
        out = l2norm(out)
        out = self.fc(self.dropout(out))

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out