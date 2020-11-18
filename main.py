import os
import pickle
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model import MPCNDataset, MPCN


def date(format='%Y-%m-%d %H:%M:%S'):
    return time.strftime(format, time.localtime())


def calculate_mse(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = [x.to(config.device) for x in batch]
            predict = model(user_reviews, item_reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # mse of dataloader


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = calculate_mse(model, train_dataloader)
    valid_mse = calculate_mse(model, valid_dataloader)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)
    best_loss, best_epoch, best_model = 1000, 0, None
    for epoch in range(config.train_epochs):
        model.train()  # turn on the train
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, item_reviews, ratings = [x.to(config.device) for x in batch]
            pred = model(user_reviews, item_reviews)
            loss = F.mse_loss(pred, ratings, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()  # summing over all loss
            total_samples += len(pred)

        model.eval()
        valid_mse = calculate_mse(model, valid_dataloader)
        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')
    return best_model


def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = calculate_mse(model, dataloader)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load word2vec and data...')

    word_emb = pickle.load(open('data/embedding/word_emb.pkl', 'rb'), encoding='iso-8859-1')
    word_dict = pickle.load(open('data/embedding/dict.pkl', 'rb'), encoding='iso-8859-1')

    train_dataset = MPCNDataset('data/music/train.csv', word_dict, config)
    valid_dataset = MPCNDataset('data/music/valid.csv', word_dict, config, retain_rui=False)
    test_dataset = MPCNDataset('data/music/test.csv', word_dict, config, retain_rui=False)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    MPCN_model = MPCN(config, word_emb, fusion_mode='sum').to(config.device)
    del train_dataset, valid_dataset, test_dataset, word_emb, word_dict

    os.makedirs('model', exist_ok=True)  # make dir if it isn't exist.
    model_Path = f'model/best_model{date("%Y%m%d_%H%M%S")}.pt'
    train(train_dlr, valid_dlr, MPCN_model, config, model_Path)
    test(test_dlr, torch.load(model_Path))
