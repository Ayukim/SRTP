import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F 
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
if BACKGROUND_AS_CLASS: 
    NUM_CLASSES += 1
    print('num_class = {0}'.format(NUM_CLASSES))

model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)
model.load_state_dict(torch.load('/Users/teawoo/Documents/SRTP/3D-UNet-main/Model/epoch9_valLoss0.05361206829547882.pth'))
model.eval()
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())