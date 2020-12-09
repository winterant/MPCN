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
        word_emb.append([0]*len(word_emb[0]))
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
        self.word_dict = word_dict
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item
        self.review_length = config.review_length
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]
        self.retain_rui = retain_rui

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating', 'reviewTime'])
        df['review'] = df['review'].apply(self._review2id)

        self.delete_idx = set()  # Save the indices of empty samples, delete them at last.
        user_reviews = self._get_reviews(df)  # Gather reviews for every user
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        retain_idx = [idx for idx in range(user_reviews.shape[0]) if idx not in self.delete_idx]
        self.user_reviews = user_reviews[retain_idx]
        self.item_reviews = item_reviews[retain_idx]
        self.rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)[retain_idx]
        del self.word_dict, self.delete_idx

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # For every sample (user,item), gather reviews for user/item.
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # Information for every user/item
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            if self.retain_rui:
                reviews = df_data['review'].to_list()  # reviews with review u for i.
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # reviews without review u for i.
            if len(reviews) < self.lowest_r_count:
                self.delete_idx.add(idx)
            reviews = self._pad_reviews(reviews)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _pad_reviews(self, reviews):
        count, length = self.review_count, self.review_length
        reviews = reviews[:count] + [[self.PAD_WORD_idx] * length] * (count - len(reviews))  # Certain count.
        reviews = [r[:length] + [0] * (length - len(r)) for r in reviews]  # Certain length of review.
        return reviews

    def _review2id(self, review):  # Split a sentence into words, and map each word to a unique number by dict.
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # word to unique number by dict.
            else:
                wids.append(self.PAD_WORD_idx)
        return wids
