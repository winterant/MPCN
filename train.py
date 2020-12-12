import os
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model import MPCN
from utils import date, calculate_mse, MPCNDataset, load_embedding


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = calculate_mse(model, train_dataloader, config.device)
    valid_mse = calculate_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)
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

        lr_sch.step()
        model.eval()
        valid_mse = calculate_mse(model, valid_dataloader, config.device)
        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')
    return best_model


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load word2vec and data...')

    word_emb, word_dict = load_embedding(config.word2vec_file)
    train_dataset = MPCNDataset(config.train_file, word_dict, config)
    valid_dataset = MPCNDataset(config.valid_file, word_dict, config, retain_rui=False)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)

    MPCN_model = MPCN(config, word_emb, fusion_mode='sum').to(config.device)
    del train_dataset, valid_dataset, word_emb, word_dict

    os.makedirs(os.path.dirname(config.saved_model), exist_ok=True)  # make dir if it isn't exist.
    train(train_dlr, valid_dlr, MPCN_model, config, config.saved_model)
