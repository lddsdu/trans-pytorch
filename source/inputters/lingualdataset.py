# -*- codingL: utf-8 -*-

import torch
import json
import tqdm
import os
from torch.utils.data import Dataset
from source.utils.consoleprint import *
from source.misc.constants import *


class LingualDataset(Dataset):
    def __init__(self, cn_corpus_filename, en_corpus_filename, cn_vocab_filename, en_vocab_filename,
                 mode="e2c", cn_max_vocab_size=30000, en_max_vocab_size=50000, debug=False, max_length=20,
                 en_vocab_t7="data/vocab.en.t7", cn_vocab_t7="data/vocab.cn.t7"):
        super(LingualDataset, self).__init__()
        self.debug = debug
        self.max_length = max_length
        self.cn_corpus_filename = cn_corpus_filename
        self.en_corpus_filename = en_corpus_filename
        self.cn_vocab_filename = cn_vocab_filename
        self.en_vocab_filename = en_vocab_filename
        self.cn_max_vocab_size = cn_max_vocab_size
        self.en_max_vocab_size = en_max_vocab_size
        self.en_vocab_t7 = en_vocab_t7
        self.cn_vocab_t7 = cn_vocab_t7
        self.en_vocab, self.cn_vocab, self.en_corpus, self.cn_corpus, self.en_lengths, self.cn_lengths\
            = self._load_data()
        consoleinfo("corpus word 2 id")
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.SOS_ID = 2
        self.EOS_ID = 3
        self.corpusstr2id()

        assert mode in ["e2c", "c2e"], "mode should be e2c or c2e, but value of {}".format(mode)
        self.english2chinese = False
        if mode == "e2c":
            self.english2chinese = True

        if self.english2chinese:
            self.data1 = self.en_corpus
            self.data2 = self.cn_corpus
            self.len1 = [min(x, max_length) for x in self.en_lengths]
            self.len2 = [min(x, max_length) for x in self.cn_lengths]
        else:
            self.data1 = self.cn_corpus
            self.data2 = self.en_corpus
            self.len1 = [min(x, max_length) for x in self.cn_lengths]
            self.len2 = [min(x, max_length) for x in self.en_lengths]

        assert len(self.en_corpus) == len(self.cn_corpus), \
            "corpus en, cn should be the same, but valued of {} vs. {}".\
            format(len(self.en_corpus), len(self.cn_corpus))

    def __getitem__(self, index):
        data1 = self.data1[index]
        data2 = self.data2[index]
        len1 = self.len1[index]
        len2 = self.len2[index]
        return data1, data2, len1, len2

    def __len__(self):
        return len(self.data1)

    def _load_data(self):
        # vocab
        def _load_vocab(vocab_filename, vocab_max_size):
            vocab = []
            with open(vocab_filename) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    items = line.split("\t")
                    if len(items) != 2:
                        continue
                    word, num = items[0], int(items[1])
                    vocab.append(word)
                    if len(vocab) >= vocab_max_size:
                        break
            return vocab
        if os.path.exists(self.cn_vocab_t7) and os.path.exists(self.en_vocab_t7):
            consoleinfo("load vocab from t7 serialize file")
            cn_vocab = torch.load(self.cn_vocab_t7)
            en_vocab = torch.load(self.en_vocab_t7)
        else:
            en_vocab = [PAD, UNK, SOS, EOS]
            cn_vocab = [PAD, UNK, SOS, EOS]
            en_vocab += _load_vocab(self.en_vocab_filename, vocab_max_size=self.en_max_vocab_size)
            en_vocab = dict(zip(en_vocab, range(len(en_vocab))))
            cn_vocab += _load_vocab(self.cn_vocab_filename, vocab_max_size=self.cn_max_vocab_size)
            cn_vocab = dict(zip(cn_vocab, range(len(cn_vocab))))

            torch.save(cn_vocab, self.cn_vocab_t7)
            torch.save(en_vocab, self.en_vocab_t7)

        # data
        def _load_corpus(corpus_filename, key):
            consoleinfo("process corpus {}".format(corpus_filename))
            with open(corpus_filename) as f:
                corpus = json.load(f)
            corpus = corpus[key]
            # 删除空格，换行符号等
            if self.debug:
                corpus = corpus[:320]
            rets = []
            for sequence in tqdm.tqdm(corpus):
                rets.append([])
                for word in sequence:
                    word = word.strip()
                    if word != "":
                        rets[-1].append(word)
            return rets
        en_corpus = _load_corpus(self.en_corpus_filename, "english")
        en_lengths = [len(x) + 1 for x in en_corpus]
        cn_corpus = _load_corpus(self.cn_corpus_filename, "chinese")
        cn_lengths = [len(x) + 1 for x in cn_corpus]
        return en_vocab, cn_vocab, en_corpus, cn_corpus, en_lengths, cn_lengths

    def corpusstr2id(self):
        def _corpusstr2id(vocab, corpus):
            temp = []
            for sequence in tqdm.tqdm(corpus):
                temp.append([self.SOS_ID])
                sequence = sequence[: self.max_length - 1]
                for word in sequence:
                    if word in vocab:
                        temp[-1].append(vocab[word])
                    else:
                        temp[-1].append(self.UNK_ID)
                temp[-1].append(self.EOS_ID)
                pad_len = self.max_length + 1 - len(temp[-1])
                for _ in range(pad_len):
                    temp[-1].append(self.PAD_ID)
            return temp

        self.en_corpus = _corpusstr2id(self.en_vocab, self.en_corpus)
        self.cn_corpus = _corpusstr2id(self.cn_vocab, self.cn_corpus)


if __name__ == '__main__':
    import os
    os.chdir("../../")
    lingual_dataset = LingualDataset("data/cn.json", "data/en.json", "data/cn_vocab.txt", "data/en_vocab.txt", debug=True)
    a, b, c, d = lingual_dataset[900]
    print(a)
    print(b)
    print(c)
    print(d)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(lingual_dataset, batch_size=10, shuffle=True)

    for i in data_loader:
        print(i)

