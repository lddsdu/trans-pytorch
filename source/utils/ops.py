import torch
import numpy as np


def one_hot(t, max_num):
    size = list(t.size())
    if size[-1] != 1:
        t = t.unsqueeze(-1)
        size = list(t.size())
    b = torch.zeros(size=size[:-1] + [max_num])
    if t.device.type == "cuda":
        b = b.to(t.device.index)
    c = b.scatter_(index=t, dim=len(size) - 1, value=1)
    return c


if __name__ == '__main__':
    print(one_hot(torch.tensor(np.random.randint(low=0, high=9, size=(10, ))).to(0), 10))
    print(one_hot(torch.tensor(np.random.randint(low=0, high=9, size=(10, 1))), 10))
