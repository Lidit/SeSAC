import numpy as np
import torch

def get_device():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    return DEVICE
def sigmoid(x):
    return 1. / (1. + np.exp(-x))