import os
import random 
import numpy as np
import torch

def set_random_seed(seeds):
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds)
    return
# set_random_seed(999)    

def set_up_gpu():
    # Set up GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      
    return device