# -*- codingL: utf-8 -*-

import torch
from torch.utils.data import Dataset


class LingualDataset(Dataset):
    def __init__(self, cn_corpus, en_corpus, cn_vocab_file, en_vocab_file):
        super(LingualDataset, self).__init__()

    def __getitem__(self, index):


