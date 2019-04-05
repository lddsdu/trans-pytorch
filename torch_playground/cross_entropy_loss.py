# -*- coding: utf-8 -*-

import torch
import numpy as np


cross_loss = torch.nn.CrossEntropyLoss()
output = torch.tensor(np.random.normal(size=(10, 20, 1000)))
output = torch.nn.Softmax(dim=-1)(output)
target = torch.tensor(np.random.randint(low=0, high=999, size=(10, 20)))

flatten_output = output.view(10 * 20, 1000)
flatten_target = target.view(10 * 20, )
loss = cross_loss(flatten_output, flatten_target)

print(loss)
