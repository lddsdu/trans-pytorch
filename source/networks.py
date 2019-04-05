# -*- coding: utf-8 -*-
import torch
import argparse
from torch.utils.data import DataLoader
from source.inputters.lingualdataset import LingualDataset
from source.models.sequence import Sequence, config
from source.misc.engine import Trainer


def parse_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    arg_cfg = parser.parse_args()
    return arg_cfg


def main():
    cfg = config()
    arg_cfg = parse_argments()
    cfg.use_gpu = torch.cuda.is_available()

    if cfg.use_gpu:
        cfg.cuda = arg_cfg.cuda
        torch.cuda.set_device(cfg.cuda)

    model = Sequence(cfg).to(cfg.cuda)
    dataset = LingualDataset("../data/cn.json", "../data/en.json",
                             "../data/cn_vocab.txt", "../data/en_vocab.txt",
                             en_vocab_t7="../data/vocab.en.t7",
                             cn_vocab_t7="../data/vocab.cn.t7",
                             debug=False)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, dataloader, dataloader,
                      num_epochs=10000, save_dir=None, valid_step=None,
                      grad_clip=None, lr_scheduler=None, save_summary=None)

    trainer.train()


if __name__ == '__main__':
    main()
