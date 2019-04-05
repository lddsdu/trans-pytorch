import torch
import numpy as np


corpus = torch.tensor(np.random.normal(size=(3, 20, 128)))
length = torch.tensor([9, 7, 3])
sorted_lengtg, indices = length.sort()
original = corpus
sorted = corpus.index_select(dim=0, index=indices)
back = sorted.index_select(dim=0, index=indices)

print(corpus.shape)
print(original.shape)
print(sorted.shape)
print(back.shape)

print(original == back)
print(original == sorted)


