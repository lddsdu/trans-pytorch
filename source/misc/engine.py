# -*- coding: utf-8 -*-
import os
import tqdm
import torch
from tensorboardX import SummaryWriter
from source.utils.consoleprint import consoleinfo


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 valid_iter,
                 train_iter,
                 num_epochs=1,
                 save_dir=None,
                 valid_step=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 save_summary=False):

        self.model = model
        self.optimizer = optimizer
        self.valid_iter = valid_iter
        self.train_iter = train_iter
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.valid_step = valid_step
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary
        self.epoch = num_epochs

        self.summary_writer = SummaryWriter(log_dir="../summary")

    def train_epoch(self, epoch, global_step):
        step = 0
        try:
            for batch_id, inputs in enumerate(tqdm.tqdm(self.train_iter), 1):
                en, cn, en_len, cn_len = inputs
                en = torch.stack(en, 0).transpose(0, 1).to(0)
                cn = torch.stack(cn, 0).transpose(0, 1).to(0)
                en_len = en_len.to(0)
                cn_len = cn_len.to(0)
                output = self.model((en, en_len), (cn, cn_len))
                loss = self.model.train(output, cn[:, 1:], self.optimizer)
                if loss is not None:
                    step += 1
                    self.summary_writer.add_scalar("scalar/loss", loss, global_step+step)

        except KeyboardInterrupt:
            self.save_model(epoch, step)
            raise Exception("finished")
        return global_step + step

    def save_model(self, epoch, step):
        model_filename = "../checkpoint/trans-e{}-s{}.t7".format(epoch, step)
        print("===> Saving Models to : \n{}".format(model_filename))
        state = {
            'state': self.model.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if not os.path.isdir("../checkpoint"):
            os.mkdir("../checkpoint")
        torch.save(state, model_filename)

    def train(self):
        global_step = 0
        save_interval = 5
        if self.train_iter.dataset.debug:
            save_interval = 1000
        for e in range(1, self.epoch + 1):
            consoleinfo("TRAINING EPOCH {}".format(e))
            global_step = self.train_epoch(e, global_step)
            consoleinfo("FINISH EPOCH {}".format(e))

            if e % save_interval == 0:
                self.save_model(e, 0)
