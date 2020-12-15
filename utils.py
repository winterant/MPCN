import os
import time
import pandas as pd

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_idx = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = word_idx
            word_idx += 1
        word_emb.append([0] * len(word_emb[0]))
        word_dict['<UNK>'] = word_idx
    return word_emb, word_dict


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def calculate_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = [x.to(device) for x in batch]
            predict = model(user_reviews, item_reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # mse of dataloader


class MPCNDataset(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True):
        self.training = ('train' in data_path)
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item
        self.review_length = config.review_length
        self.PAD_idx = word_dict[config.PAD_WORD]

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating', 'reviewTime'])
        df['review'] = df['review'].apply(lambda r: [word_dict.get(w, self.PAD_idx) for w in str(r).split()])
        if not self.training:
            df1 = pd.read_csv(os.path.dirname(data_path) + '/valid_test.csv',
                              header=None, names=['userID', 'itemID', 'review', 'rating', 'reviewTime'])
            df1['review'] = df1['review'].apply(lambda r: [word_dict.get(w, self.PAD_idx) for w in str(r).split()])
            self.non_train_df = df1

        user_reviews, del_idx_u = self._get_reviews(df, retain_rui=retain_rui)  # Gather reviews for every user
        item_reviews, del_idx_i = self._get_reviews(df, 'itemID', 'userID', retain_rui)
        retain_idx = [idx for idx in range(user_reviews.shape[0]) if idx not in (del_idx_u | del_idx_i)]
        self.user_reviews = user_reviews[retain_idx]
        self.item_reviews = item_reviews[retain_idx]
        self.rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)[retain_idx]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, groupBy='userID', target='itemID', retain_rui=True):
        # For every sample (user,item), gather reviews for user or item.
        if self.training:
            reviews_groups = dict(list(df[[target, 'review']].groupby(df[groupBy])))
        else:
            reviews_groups = dict(list(self.non_train_df[[target, 'review']].groupby(self.non_train_df[groupBy])))
        group_reviews = []
        del_idx = set()  # gather indices which review has no word
        for idx, (group_id, target_id) in enumerate(zip(df[groupBy], df[target])):
            df_data = reviews_groups[group_id]  # group
            if retain_rui:
                reviews = df_data['review'].to_list()  # reviews with review u for i.
            else:
                reviews = df_data['review'][df_data[target] != target_id].to_list()  # reviews without review u for i.
            if len(reviews) < self.lowest_r_count:
                del_idx.add(idx)
            reviews = self._pad_reviews(reviews)
            group_reviews.append(reviews)
        return torch.LongTensor(group_reviews), del_idx

    def _pad_reviews(self, reviews):
        count, length = self.review_count, self.review_length
        reviews = reviews[:count] + [[self.PAD_idx] * length] * (count - len(reviews))  # Certain count.
        reviews = [r[:length] + [0] * (length - len(r)) for r in reviews]  # Certain length of review.
        return reviews
