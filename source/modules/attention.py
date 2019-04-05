# -*- coding: utf-8 -*-

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.query_size = config.query_size
        self.memory_size = config.memory_size or self.query_size
        self.hidden_size = config.hidden_size or self.query_size
        self.mode = config.attention_mode

        if self.mode == "general":
            self.linear_query = torch.nn.Linear(self.query_size, self.memory_size, bias=False)
        elif self.mode == "perceptual":
            self.linear_query = torch.nn.Linear(self.query_size, self.hidden_size, bias=True)
            self.linear_memory = torch.nn.Linear(self.memory_size, self.hidden_size, bias=False)
            self.tanh = torch.nn.Tanh()
            self.v = torch.nn.Linear(self.hidden_size, 1, bias=False)
        elif self.mode == "dot":
            pass
        else:
            raise NotImplementedError("wrong mode")

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, memory, mask=None):
        """forward attention

        # Args
            query: bs, query_embedding_dim
            memory: bs, memory_num, memory_embedding_dim
            mask: bs, memory_num
        """
        if self.mode == "dot":
            assert query.size(2) == memory.size(-1)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            key = self.linear_query(query)
            attn = torch.bmm(key, memory.transpose(1, 2))
        else:
            # bs, memory_num, hidden_size
            hidden = self.linear_query(query).unsqueeze(1) + self.linear_memory(memory)
            key = self.tanh(hidden)
            # bs, memory_num
            attn = self.v(key).squeeze(-1)

        if mask is not None:
            # bs, query_length, memory_length
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.mask_fill(mask, -float("inf"))

        weights = self.softmax(attn).unsqueeze(1)
        # bs, 1, memory_num  *  bs, memory_num, memory_dim -> bs, memory_dim
        weighted_memory = torch.bmm(weights, memory)

        return weights, weighted_memory
