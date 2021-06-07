import os

import torch

RESULTS_DIR = os.environ['PSQI_RESULTS_DIR']
DATA_DIR = os.environ['PSQI_DATA_DIR']
DEVICE_NAME = os.environ.get('PSQI_DEVICE', 'cpu')
DEVICE = torch.device(DEVICE_NAME)
IS_CUDA = DEVICE_NAME.startswith('cuda')

print('Data directory %s' % DATA_DIR)
print('Results directory %s' % RESULTS_DIR)
print('Using gpu - %s' % ('yes' if IS_CUDA else 'no'))
print('Device "%s" used for computations' % DEVICE_NAME)
