import torch
import numpy as np


def main():
    # a = torch.tensor([[1, 2, 3]], dtype=torch.float32).to(0)
    # b = torch.tensor([[1], [2], [3]], dtype=torch.float32).to(0)
    # c = a * b
    # print(c.device)
    # print(c)
    device = 0
    gru = torch.nn.GRU(input_size=10,
                       hidden_size=20,
                       dropout=0.2,
                       bidirectional=True,
                       num_layers=2, batch_first=True).to(device)

    hidden_states = torch.tensor(np.random.normal(size=(4, 3, 20)), dtype=torch.float32).to(device)   # (4) layer * bi
    inputs = torch.tensor(np.random.normal(size=(3, 20, 10)), dtype=torch.float32).to(device)
    output_layer = torch.nn.Linear(40, 100).to(device)
    print(inputs)
    outputs, states = gru(inputs, hidden_states)

    print(outputs.shape)
    print(states.shape)

    probs = output_layer(outputs)
    print(probs)

main()