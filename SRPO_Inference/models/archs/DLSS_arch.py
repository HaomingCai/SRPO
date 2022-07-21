import torch.nn as nn
import models.archs.arch_util as arch_util
import torch
import torch.nn.functional as F
import torchvision


'''
Due to commercial confidentiality agreement, we cannot release the specific design of SRPO from code aspect.
However, based on paper and pretrained pth file, you can easily reproduce our SRPO. 
'''

class SRPO(nn.Module):
    def __init__(self, input_channel=3, l1_c=16, l1_k=5, l2_c=16, l2_k=5, l3_c=2, l3_k=5, flat_sr_up_type='bilinear', offset_up_type='bilinear'):
        super(SRPO, self).__init__()

        # Components of SRPO
        
    def forward(self, x):
        # x is the input image with simple data augmentations. Check LQGT_dataset.py.
        
        # Go through the SRPO

        # definition of offset_sr and offset can be found in our paper
        output = torch.cat([offset_sr, offset], dim=1)
        return None, output




