import torch
import random
import numpy as np

def config_seed(SEED) :
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)