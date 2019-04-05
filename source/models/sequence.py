# -*- coding: utf-8 -*-

import torch
from source.modules.embedder import Embedder
from source.modules.rnnencoder import RNNEncoder
from source.modules.rnndecoder import RNNDecoder
from source.misc import config as cfg


def config():
    config = cfg.Config()
    config.add(input_size=128)
    config.add(hidden_size=128)
    config.add(beam_width=3)
    config.add(rnn_hidden_size=256)
    config.add(bi=True)
    config.add(max_decode_step=20)
    config.add(query_size=128)
    config.add(memory_size=128)
    config.add(hidden_size=128)
    config.add(mode="e2c")
    config.add(hidden_units=128)
    config.add(num_layers=2)
    config.add(dropout=0.2)
    config.add(attention_mode="perceptual")
    config.add(en_vocab_size=50000)
    config.add(cn_vocab_size=30000)
    config.add(output_size=config.en_vocab_size)
    config.add(is_train=True)
    config.add(embedding_dim=128)
    config.add(embedder=None)
    config.add(gpu=0)
    config.add(batch_size=32)
    # TODO a lot of parameters need to add
    return config


class Sequence(torch.nn.Module):
    def __init__(self, config_):
        super(Sequence, self).__init__()
        self.config = config_
        self.is_train = self.config.is_train

        # input_size = self.config.input_size
        # hidden_size = self.config.hidden_size
        # rnn_hidden_size = self.config.rnn_hidden_size

        self.beam_width = self.config.beam_width
        self.max_decode_step = self.config.max_decode_step
        # modules
        self.en_embedder = Embedder(num_embeddings=self.config.en_vocab_size + 4,
                                    embedding_dim=self.config.embedding_dim).to(self.config.gpu)
        self.cn_embedder = Embedder(num_embeddings=self.config.cn_vocab_size + 4,
                                    embedding_dim=self.config.embedding_dim).to(self.config.gpu)

        self.encoder = RNNEncoder(config_)
        self.decoder = RNNDecoder(config_)

        self.softmax = torch.nn.Softmax(dim=-1)
        # loss two input sized of bs, s_len, n_class and bs, s_len
        self.nll_loss = torch.nn.NLLLoss(
            weight=torch.tensor([0] + [1] * (self.config.cn_vocab_size + 3), dtype=torch.float32)).to(self.config.gpu)

    def forward(self, enc_inputs, dec_inputs, hidden=None, is_train=None):
        if is_train is None:
            is_train = self.is_train
        # enc_outputs: sequence_length, batch_size, hidden_units
        # enc_last_hidden: layer, batch_size, hidden_units
        enc_inputs = (self.en_embedder(enc_inputs[0]), enc_inputs[1])
        enc_outputs, enc_last_hidden = self.encoder(enc_inputs, hidden=None)

        if is_train:
            assert dec_inputs is not None
            # target = dec_inputs[0][:, 1:].contiguous()
            dec_inputs = (self.cn_embedder(dec_inputs[0][:, :-1]), dec_inputs[1])
            probs = self.decoder(dec_inputs, enc_outputs, enc_last_hidden)
            return probs
        else:
            # output
            output_int = torch.zeros(shape=(enc_inputs.size(0), self.max_decode_step))
            # input single step
            last_state = enc_last_hidden
            memory = enc_outputs
            inputs = torch.ones(shape=(enc_inputs.size(0), )).unsqueeze(1)
            # no beam search
            for i in range(self.max_decode_step):
                inputs_embeded = self.cn_embedder(inputs)
                vocab_prob_output, output, last_state = \
                    self.decoder.decode_single_step(inputs_embeded, memory, last_state, is_train=False)
                output_int[:, i] = torch.argmax(vocab_prob_output, dim=0)

            return output_int

    def train(self, output, target, optimizer):
        if output.size(0) != self.config.batch_size:
            return
        flatten_output = output.view(self.config.batch_size * 20, -1)
        flatten_target = target.contiguous().view(self.config.batch_size * 20, )
        # one_hot_target = one_hot(target, self.config.cn_vocab_size + 4)
        optimizer.zero_grad()
        loss = self.nll_loss(flatten_output, flatten_target)
        loss.backward()
        optimizer.step()
        return loss
