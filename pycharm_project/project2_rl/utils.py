import numpy as np
import torch
import time
import datetime

# Date Time Format
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"

def get_today_str():
    today = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime(FORMAT_DATE)
    return today_str

def get_time_str():
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)
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
