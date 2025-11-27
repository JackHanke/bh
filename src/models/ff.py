import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class FF(nn.Module):
    def __init__(self, dim):
        super(VAE, self).__init__()
        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.net(x)

        return y