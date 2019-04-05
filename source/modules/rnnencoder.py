from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch


class RNNEncoder(nn.Module):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config
        num_direction = 2 if self.config.bi else 1

        # 参数定义了GRU的 1.input_size, hidden_size, num_layers,
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.rnn_hidden_size = (config.hidden_size // num_direction)
        self.embedder = config.embedder
        self.num_layers = config.num_layers
        self.bi = config.bi
        self.dropout = config.dropout

        assert self.hidden_size % num_direction == 0, "wrong hidden size and bi"

        # 两层双向GRU
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bi)

    def forward(self, inputs, hidden=None):
        """forward

        # Args
            inputs: A tensor(bs, sequence_length, embedding_dim) [A tensor(bs,)]
            hidden: A tensor(num_layers * bi, bs, hidden_units) or None
        # Returns
            outputs: A tensor(bs, sequence_length, hidden_units)
            last_hidden:A tensor(num_layers, batch_size, num_directions * hidden_size)
        """
        # inputs: bs, length
        # lengths:bs,
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None
        # rnn_inputs: bs, length, embedding_dim
        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs
        # bs, sequence_length, dim_embedding
        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)
            # sorted_lengths.sum().item(), dim_embedding
            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        # RNN process
        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        if self.bi:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            # bs, sequence_length, hidden_units
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                # 对于没有处理的sample直接使用zero来填充
                zeros = outputs.new_zeros(batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(self.num_layers, batch_size - num_valid, self.hidden_size)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            # num_layers, bs, num_directions * hidden_size
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    def _bridge_bidirectional_hidden(self, hidden):
        """
        bidirection hidden is shaped of (num_layers * num_dicrections, batch_size, hidden_size)
        need to change it to (num_layers, batch_size, num_dictections * hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)