import pickle
import time
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import date, calculate_mse, MPCNDataset, load_embedding


def test(dataloader, model, device):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = calculate_mse(model, dataloader, device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load word2vec and test data...')

    word_emb, word_dict = load_embedding(config.word2vec_file)
    test_dataset = MPCNDataset(config.test_file, word_dict, config, retain_rui=False)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    MPCN_model = torch.load(config.saved_model)
    test(test_dlr, MPCN_model, config.device)
