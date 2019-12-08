import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMapHook(nn.Module):
    def __init__(self, detach, device):
        """
            detach = true or false
        """
        super(FeatureMapHook, self).__init__()
        self.detach = detach
        self.input = None
        self.device = device
    def forward(self, input):
        self.input = input
        return input

    def get_feature_map(self):
        if self.detach:
            return self.input.detach()
        else:
            return self.input

class ContentLoss(nn.Module):
    def __init__(self, device):
        super(ContentLoss, self).__init__()
        self.device = device

    def update(self, content_map):
        self.content_map = content_map.detach()

    def forward(self, syn_map):
        return F.mse_loss(syn_map, self.content_map)


class MRFStyleLoss(nn.Module):
    def __init__(self, style_map, patch_size, device,gpu_chunk_size=256, style_stride = 2, syn_stide = 2):
        super(MRFStyleLoss, self).__init__()
        self.patch_size = patch_size
        self.device = device
        # don't forget to initialze these for the first time !!!
        self.syn_stride = 2
        self.style_stride = 2
        self.style_patches = []
        self.style_patches_norm = []
        self.gpu_chunk_size = gpu_chunk_size

    def update(self, style_map):
        #do update here
        self.style_patches = self.get_patches(style_map.detach(), self.patch_size, self.style_stride)
        self.style_patches_norm = self.frob_norm().view(-1, 1, 1).to(self.device)
        print(self.device)
        # print(self.style_patches_norm.device.type)

    def frob_norm(self):
        norms = torch.zeros(self.style_patches.shape[0])
        for i in range(len(self.style_patches)):
            norms[i] = (torch.sum(self.style_patches[i] ** 2)) ** 0.5

        return norms.to(self.device)

    def get_patches(self, img, patch_size, stride):
        H,W = img.shape[2], img.shape[3]
        patches = []
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = img[:, :, i:i + patch_size, j : j + patch_size]
                patches.append(patch)
        patches = torch.cat(patches, dim=0).to(self.device)
        return patches

    def forward(self, syn_map):
        loss = 0.0
        syn_patches = self.get_patches(syn_map, self.patch_size, self.syn_stride)
        max_response = []

        for i in range(0, len(self.style_patches), self.gpu_chunk_size):
            start, end = i, min(i + self.gpu_chunk_size, len(self.style_patches))
            weights = self.style_patches[start : end]
            response = F.conv2d(syn_map, weights, stride=self.syn_stride)
            max_response.append(response.squeeze(0))


        max_response = torch.cat(max_response, dim=0)
        max_response = max_response.div(self.style_patches_norm)
        max_response = torch.argmax(max_response, dim=0).view(1, -1).squeeze()
        # max_response = torch.argmax(max_response, dim=0)
        # max_response = torch.reshape(max_response, (1, -1)).squeeze()

        for i in range(0, len(max_response), self.gpu_chunk_size):
            start, end = i, min(i + self.gpu_chunk_size, len(max_response))
            syn_idx = tuple(range(start, end))
            style_idx = tuple(max_response[start : end])
            loss += torch.sum(torch.mean((syn_patches[syn_idx,:,:,:] - self.style_patches[style_idx,:,:,:]) ** 2, dim = [1,2,3]))
        loss /= len(max_response)

        return loss



class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, input):
        img = input.squeeze().permute([1, 2, 0])
        r = (img[:, :, 0] + 2.12) / 4.37
        g = (img[:, :, 1] + 2.04) / 4.46
        b = (img[:, :, 2] + 1.80) / 4.44

        temp = torch.cat([r.unsqueeze(2), g.unsqueeze(2), b.unsqueeze(2)], dim=2)
        gx = torch.cat((temp[1:, :, :], temp[-1, :, :].unsqueeze(0)), dim=0)
        gx = gx - temp

        gy = torch.cat((temp[:, 1:, :], temp[:, -1, :].unsqueeze(1)), dim=1)
        gy = gy - temp

        return torch.mean(torch.pow(gx, 2)) + torch.mean(torch.pow(gy, 2))
