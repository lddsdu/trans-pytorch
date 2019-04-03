# -*- coding: utf-8 -*-

import jieba
import tqdm
import json
import argparse
from utils import read_corpus


def tokenize(sequence):
    word_list = jieba.lcut(sequence)
    return word_list


def vocabize(corpus):
    vocab = {}
    for sequence in corpus:
        for word in sequence:
            if word not in vocab:
                vocab[word] = 1
            vocab[word] = vocab[word] + 1
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chinese_corpus", type=str, default="data/neu2017/NEU_cn.txt")
    parser.add_argument("--target", type=str, default="data/cn.json")
    parser.add_argument("--vocab_file", type=str, default="data/cn_vocab.txt")

    config = parser.parse_args()
    cn_corpus = config.chinese_corpus
    target_filename = config.target
    vocab_filename = config.vocab_file
    cn_corpus = read_corpus(cn_corpus)
    temp = []
    for sequence in tqdm.tqdm(cn_corpus):
        temp.append(tokenize(sequence))
    cn_corpus = temp
    cn_vocab = vocabize(cn_corpus)

    corpus = {"chinese": cn_corpus}
    corpus_str = json.dumps(corpus)
    with open(target_filename, "w") as f:
        f.write(corpus_str)

    vocab_str = "\n".join(map(lambda x: "{}\t{}".format(x[0], x[1]),
                              sorted(cn_vocab.items(), key=lambda x: x[1], reverse=True)))

    with open(vocab_filename, "w") as f:
        f.write(vocab_str)
    print("Done")


if __name__ == "__main__":
    main()

