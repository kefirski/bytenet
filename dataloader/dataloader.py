import collections
import os

import numpy as np
import torch as t
from six.moves import cPickle
from torch.autograd import Variable
from .beam import Beam


class Dataloader():
    def __init__(self, data_path='', force_preprocessing=False):
        """
        :param data_path: path to data
        :param force_preprocessing: whether to preprocess data even if it was preprocessed before
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.data_path = data_path
        self.prep_path = self.data_path + 'preprocessings/'

        if not os.path.exists(self.prep_path):
            os.makedirs(self.prep_path)

        '''
        go_token (stop_token) uses to mark start (end) of the sequence
        pad_token uses to fill tensor to fixed-size length

        In order to make model work correctly,
        these tokens should be unique
        '''
        self.go_token = '∑'
        self.pad_token = 'Œ'
        self.stop_token = '∂'

        self.data_files = self.en_ru_dir(data_path + 'en.txt', data_path + 'ru.txt', )

        self.idx_file = self.prep_path + 'vocab.pkl'
        self.tensor_file = self.prep_path + 'tensor.pkl'

        idx_exists = os.path.exists(self.idx_file)
        tensor_exists = os.path.exists(self.tensor_file)

        preprocessings_exist = all([file for file in [idx_exists, tensor_exists]])

        if preprocessings_exist and not force_preprocessing:
            print('Loading preprocessed data have started')
            self.load_preprocessed()
            print('Preprocessed data have loaded')
        else:
            print('Processing have started')
            self.preprocess()
            print('Data have preprocessed')

    def build_vocab(self, sentences):
        """
        :param sentences: An array of chars in data
        :return:
            vocab_size – Number of unique words in corpus
            idx_to_token – Array of shape [vocab_size] containing list of unique chars
            token_to_idx – Dictionary of shape [vocab_size]
                such that idx_to_token[token_to_idx[some_char]] = some_char
                where some_char is is from idx_to_token
        """

        char_counts = collections.Counter(sentences)

        idx_to_token = [x[0] for x in char_counts.most_common()]
        idx_to_token = [self.pad_token, self.go_token, self.stop_token] + list(sorted(idx_to_token))

        token_to_idx = {x: i for i, x in enumerate(idx_to_token)}

        vocab_size = len(idx_to_token)

        return vocab_size, idx_to_token, token_to_idx

    @staticmethod
    def en_ru_dir(en, ru):
        return {'en': en, 'ru': ru}

    def preprocess(self):

        data = [open(path, "r").read().lower() for path in self.data_files.values()]

        self.vocab_size, self.idx_to_token, self.token_to_idx = self.build_vocab(data[0] + data[1])

        data = [target.split('\n')[:-1] for target in data]
        data = [[[self.token_to_idx[token]
                  for token in self.go_token + line + self.stop_token]
                 for line in target]
                for target in data]
        self.data = self.en_ru_dir(data[0], data[1])
        del data

        self.max_len = max([
            max([
                len(line) for line in lines
            ])
            for lines in self.data.values()
        ])

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_token, f)

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

    def load_preprocessed(self):

        self.idx_to_token = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_token)
        self.token_to_idx = dict(zip(self.idx_to_token, range(self.vocab_size)))

        self.data = cPickle.load(open(self.tensor_file, "rb"))

        self.max_len = max([
            max([
                len(line) for line in lines
            ])
            for lines in self.data.values()
        ])

    def next_batch(self, batch_size):
        """
        :param batch_size: number of selected data elements
        :return: target tensors
        """

        indexes = np.array(np.random.randint(len(self.data['en']), size=batch_size))
        lines = [[self.data[target][index] for index in indexes] for target in ['en', 'ru']]

        return self.construct_batches(lines)

    def construct_batches(self, lines):
        """
        :param lines: An list of indexes arrays
        :return: Batches
        """

        condition = lines[0]
        input = [line[:-1] for line in lines[1]]
        target = [line[1:] for line in lines[1]]

        condition = self.padd_sequences(condition)
        input = self.padd_sequences(input)
        target = self.padd_sequences(target)

        return condition, input, target

    @staticmethod
    def padd_sequences(lines):

        lengths = [len(line) for line in lines]
        max_length = max(lengths)
        max_length = max_length if max_length > 20 else 20

        # Pad token has idx 0
        return np.array([line + [0] * (max_length - lengths[i])
                         for i, line in enumerate(lines)])

    def torch(self, batch_size, cuda, volatile=False):

        condition, input, target = self.next_batch(batch_size)
        condition, input, target = [Variable(t.from_numpy(var), volatile=volatile)
                                    for var in [condition, input, target]]

        if cuda:
            condition, input, target = [var.cuda() for var in [condition, input, target]]

        return condition, input, target

    def go_input(self, batch_size, cuda):

        go_input = np.array([[self.token_to_idx[self.go_token]]] * batch_size)
        go_input = Variable(t.from_numpy(go_input)).long()

        if cuda:
            go_input = go_input.cuda()

        return go_input

    def to_tensor(self, lines, cuda, volatile=True):

        tensor = Variable(t.LongTensor([[self.token_to_idx[self.go_token]] +
                                        [self.token_to_idx[token] for token in line]
                                        for line in lines]), volatile=volatile)
        if cuda:
            tensor = tensor.cuda()

        return tensor

    def sample_char(self, probs, n_beams):

        probs = [[i, val] for i, val in enumerate(probs)]
        probs = sorted(probs, key=lambda pair: pair[1])[-n_beams:]

        return [Beam(p, self.idx_to_token[idx]) for idx, p in probs]

    def beam_update(self, beams, probs):

        n_beams = len(beams)

        for i in range(n_beams):
            probs[i] *= beams[i].prob

        probs = [[beam, idx, p] for i, beam in enumerate(beams) for idx, p in enumerate(probs[i])]
        probs = sorted(probs, key=lambda triple: triple[2])[-n_beams:]
        return [beam.update(prob, self.idx_to_token[idx]) for beam, idx, prob in probs]
