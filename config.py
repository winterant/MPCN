import argparse
import inspect

import torch


class Config:
    # file path
    word2vec_file = 'embedding/glove.6B.100d.txt'
    train_file = 'data/music/train.csv'
    valid_file = 'data/music/valid.csv'
    test_file = 'data/music/test.csv'
    saved_model = 'model/best_model.pt'

    # attribute of dataset
    review_count = 10  # max count of reviews for every user or item
    review_length = 30  # max count of review words
    lowest_review_count = 2  # Minimum number of comments for users to keep
    PAD_WORD = '<UNK>'

    # running device and hyper parameter of training
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    train_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    learning_rate_decay = 0.99
    l2_regularization = 1e-6
    pointer_count = 2
    fm_hidden = 10  # Hidden dim of Factorization Machine

    # auto argparse
    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)
