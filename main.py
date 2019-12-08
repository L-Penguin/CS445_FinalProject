import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import cv2
import os
from torch.autograd import Variable

from model import MRFCNN
from loss_libs import ContentLoss, MRFStyleLoss, TVLoss


################## parameter list, to be adjusted
num_res = 3
style_weight = 0.4
tv_weight = 0.1
gpu_chunck = 256
style_stride = 2
syn_stride = 2
sample_step = 50
content_path = "data/content1.jpg"
style_path =  "data/style1.jpg"
max_iter = 100
iter = 0

def get_transforms():
    """

    """
    normalizer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    denormalizer = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))
    return normalizer, denormalizer

def upsample(H, W, img, device):
    img = F.interpolate(img, size=[H, W], mode="bilinear")
    img = img.clone().detach().requires_grad_(True).to(device)
    return img

def downsample(scale, img):
    return F.interpolate(img, scale_factor=scale, mode='bilinear')


def main():
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # transforms to preprocess and recover images
    normalizer, denormalizer = get_transforms()

    content = cv2.imread(content_path)
    content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
    content = normalizer(content).unsqueeze(0).to(device)

    style = cv2.imread(style_path)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    style = normalizer(style).unsqueeze(0).to(device)

    # create image pyramid with bilinear interpolation
    pyr_content, pyr_style = [], []
    for i in range(num_res):
        scale = 0.5 ** (num_res - 1 - i)
        pyr_content.append(downsample(scale, content))
        pyr_style.append(downsample(scale, style))

    # initialze net with the lowest resolution pair of images
    net = MRFCNN(pyr_content[0], pyr_style[0], style_weight, tv_weight, style_stride, syn_stride, gpu_chunck, device).to(device)

    synthesis = None
    net.train()
    global iter
    iter = 0
    for i in range(num_res):
        print("working on res {}".format(i))
        if i == 0:
            synthesis = pyr_content[0].clone().requires_grad_(True).to(device)

        else:
            H, W = pyr_content[i].shape[2:4]
            synthesis = upsample(H, W, synthesis, device)
            net.update_content_and_style_img(pyr_content[i], pyr_style[i])

        optimizer = optim.LBFGS([synthesis], lr=1, max_iter=100)

        def closure():
            global iter
            optimizer.zero_grad()
            loss = net(synthesis)
            loss.backward(retain_graph = True)
            print(loss.item())
            print(iter)
            if (iter + 1) % 10 == 0:
                print('res-%d-iteration-%d: %f' % (i+1, iter + 1, loss.item()))
            # save image
            if (iter + 1) % sample_step == 0 or iter + 1 == max_iter:
                img = denormalizer(synthesis.clone().squeeze().to(torch.device("cpu"))).unsqueeze(0)
                img = F.interpolate(img, size=content.shape[2:4], mode="bilinear")
                torchvision.utils.save_image(img.squeeze(), 'res-%d-result-%d.jpg' % (i+1, iter + 1))
                print("yeah!!!")

            iter += 1
            if iter == max_iter:
                iter = 0
            return loss
        for s in range(max_iter):
            optimizer.step(closure)

if __name__ == "__main__":
    main()
