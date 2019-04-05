from torch.nn import Embedding
import torch


class Embedder(Embedding):

    def load_embedding(self, embeds, scale=0.05):
        assert len(embeds) == self.num_embeddings
        embeds = torch.tensor(embeds, dtype=torch.float32)
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                torch.nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))
