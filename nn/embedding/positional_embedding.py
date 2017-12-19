import numpy as np
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal


class PositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_size):
        super(PositionalEmbeddings, self).__init__()

        self.max_len = max_len
        self.embedding_size = embedding_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.positional_embeddings = nn.Embedding(int(max_len), embedding_size, padding_idx=0)

        self.embeddings.weight = xavier_normal(self.embeddings.weight)
        self.embeddings.weight.data[0].fill_(0)
        self.position_encoding_init()

    def forward(self, input):
        batch_size, seq_len = input.size()

        positional = Variable(t.LongTensor([i for i in range(1, seq_len + 1)])).repeat(batch_size).view(batch_size, -1)
        if input.is_cuda:
            positional = positional.cuda()

        padding_mask = t.eq(input, 0).data
        positional.data.masked_fill_(padding_mask, 0)

        return self.embeddings(input) + self.positional_embeddings(positional)

    def position_encoding_init(self):
        encoding = np.array([
            [pos / np.power(10000, 2 * i / self.embedding_size) for i in range(self.embedding_size)]
            if pos != 0 else np.zeros(self.embedding_size) for pos in range(self.max_len)])

        encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
        encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])

        self.positional_embeddings.weight = nn.Parameter(t.from_numpy(encoding).float(), requires_grad=False)
