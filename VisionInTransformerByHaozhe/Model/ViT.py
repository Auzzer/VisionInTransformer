from Model.Transformer import Transformer
import torch.nn as nn
class Vit(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_classes: int,
                 dim: int, depth: int, heads: int, mlp_dim: int, channels=3,
                 dropout=0., emb_dropout=0.):
        super(Vit, self).__init__()
