import torch
import torch.nn as nn
import torch.nn.functional as functional

class FeatureMapHook(nn.Module):
    def __init__(self, detach, device):
        """
            detach = true or false
        """
        super(FeatureMapHook, self).__init__()
        self.input = None
        self.device = device
    def forward(self, input):
        self.input = input
        return input

    def get_feature_map(self):
        if detach:
            return self.input.detach()
        else:
            return self.input

class ContentLoss():
    def __init__(self, device):
        super(ContentLoss, self).__init__()
        self.device = device

    def forward(self, feature_map, target):
        loss = 0.0
        return loss

class MRFStyleLoss(feature_map, target):
    def __init__(self, target, patch_size, gpu_chunk_size=256, device):
        super(MRFStyleLoss, self).__init__()
        self.patch_size = patch_size
        self.device = device

        # don't forget to initialze these for the first time !!!
        self.patches = []
        self.patches_norm = []
        self.gpu_chunk_size = gpu_chunk_size

    def update(self, new_feature_map):
        #do update here

    def forward(self, feature_map, target):
        loss = 0.0
        return loss

class TVLoss(input):
    def __init__(self):
        super(TVLoss, self).__init__()
    def forward(self, input):
        loss = 0.0
        return loss


# class ContentLoss(_Loss):
#     def __init__(self, device):
#         super(MRFStyleLoss, self).__init__()
#         self.device = device
#
#     def forward(self, feature_map, target):
#         loss = 0.0
#         return loss
#
# class MRFStyleLoss(_Loss):
#     def __init__(self, patch_size, device):
#         """
#         inputs:
#             patch_size = (N x N)
#             device = device
#         """
#         super(MRFStyleLoss, self).__init__()
#         self.patch_size = patch_size
#         self.device = device
#
#     # remember this is not default function
#     def update(self, input):
#         return X
#
#     def forward(self, feature_map, target):
#         loss = None
#         return loss
#
# class TVLoss(nn.Module):
#     def __init__(self):
#         super()
