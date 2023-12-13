import torch
import numpy as np
import random

def random_seeder(seed):
    """Fix randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False