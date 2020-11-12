import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    train_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    learning_rate_decay = 0.99
    l2_regularization = 1e-6

    review_count = 10  # max count of reviews for every user or item
    review_length = 30  # max count of review words
    lowest_review_count = 2  # Minimum number of comments for users to keep
    PAD_WORD = '<UNK>'
    pointer_count = 2
    fm_hidden = 10  # Hidden dim of Factorization Machine
