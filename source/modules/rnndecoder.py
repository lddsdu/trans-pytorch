# -*- coding: utf-8 -*-

import torch
from torch import nn
from source.modules.attention import Attention


class RNNDecoder(nn.Module):
    def __init__(self, config, is_train=True):
        super(RNNDecoder, self).__init__()
        self.config = config
        self.is_train = is_train
        self.input_size = self.config.input_size
        self.output_size = self.config.cn_vocab_size + 4
        self.hidden_units = self.config.hidden_units
        self.num_layers = self.config.num_layers
        self.dropout = self.config.dropout
        self.embedder = self.config.embedder
        self.memory_size = self.config.memory_size or self.config.hidden_units

        self.attention_mode = self.config.attention_mode
        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_units

        if self.attention_mode:
            self.attention = Attention(config)
            self.rnn_input_size += self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_units,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0)

        self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_units),
                nn.Linear(self.hidden_units, self.output_size),
                nn.LogSoftmax(dim=-1))

    def sequence_mask(self, max_lengths, max_len=None):
        """sequence_mask

        # Args
            max_lengths: bs,
        """
        if max_len is None:
            max_len = max_lengths.max().item()
        mask = torch.arange(0, max_len, dtype=torch.long).type_as(max_lengths)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(1, *max_lengths.size(), 1)
        mask = mask.squeeze(0)
        mask = mask.lt(max_lengths.unsqueeze(-1))
        return mask

    def decode_single_step(self, input, memory, state, is_train=True):
        """decode step

        # Args:
            input: num_valid, embedding_dim
            memory: num_valid, enc_sequence_length, hidden_units
            state: 2, m_valid, hidden_units

        # Returns:
            output: num_valid, hidden_units
            new_hidden: num_valid, hidden_units
        """
        if self.embedder is not None:
            input = self.embedder(input)

        # bs, 1, embedding_dim
        input = input.unsqueeze(1)

        if self.attention_mode is not None:
            weight, weighted_memory = self.attention(state[1], memory)
            input = torch.cat([input, weighted_memory], dim=-1)

        output, new_hidden = self.rnn(input, state)

        if is_train:
            return output, new_hidden

        vocab_prob_output = self.output_layer(output)
        return vocab_prob_output, output, new_hidden

    def forward(self, inputs, memory, state):
        """forward

        # Args
            inputs: A tensor(bs, sequence_length, embedding_dim)
            memory: A tensor(bs, enc_max_length, hidden_units)
            state: A tensor(num_layers, bs, hidden_units)
        # Returns
            probs: A tensor(bs, sequence_length, self.output_size(vocab_size))
            state: A tensor(num_layers, bs, hidden_size)
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size(0), inputs.size(1)

        # rnn输出logit, (bs, sequence_length, out_input_size)来表示
        rnn_out_inputs = inputs.new_zeros(size=(batch_size, max_len, self.hidden_units), dtype=torch.float32)
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(1, indices)
        memory = memory.index_select(0, indices)

        num_valid_list = self.sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[: num_valid, i]
            valid_state = state[:, :num_valid, :].contiguous()   # 选择第二层作为输入
            rnn_out_input, hidden = self.decode_single_step(dec_input, memory[: num_valid], valid_state, is_train=self.is_train)
            state = hidden
            rnn_out_inputs[:num_valid, i:i+1, :] = rnn_out_input

        _, inv_indices = indices.sort()
        rnn_out_inputs = rnn_out_inputs.index_select(0, inv_indices)
        probs = self.output_layer(rnn_out_inputs)
        return probs
