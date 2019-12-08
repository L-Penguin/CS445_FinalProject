import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import cv2
import os


################## parameter list, to be adjusted
num_res = 3
style_weight = 0.4
tv_weight = 0.1
gpu_chunck = 256
style_stride = 2
syn_stride = 2
sample_step = 50
content_path = None
style_path =  None
max_iter = 100

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

    # transforms to preprocess and recover images
    normalizer, denormalizer = get_transforms()

    content = cv2.imread(content_path)
    content = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    content = normalizer(content_image).unsqueeze(0).to(device)

    style = cv2.imread(style_path)
    style = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    style = normalizer(style_image).unsqueeze(0).to(device)

    # create image pyramid with bilinear interpolation
    pyr_content, pyr_style = [], []
    for i in range(num_res):
        scale = 0.5 ** (num_res - 1 - i)
        pyr_content.append(downsample(scale, content))
        pyr_style.append(tmp_style = downsample(scale, style))

    # initialze net with the lowest resolution pair of images
    net = MRFCNN(pyr_content[0], pyr_style[0], style_weight, tv_weight, style_stride, syn_stride, gpu_chunck, device).to(device)

    synthesis = None
    net.train()
    for i in range(num_res):
        if i == 0:
            synthesis = pyr_content[0].clone().requires_grad_(True).to(device)

        else:
            H, W = pyr_content[i].shape
            synthesis = upsample(H, W, synthesis, device)
            net.update_style_and_content_image(pyr_content[i], pyr_style[i])

        optimizer = optim.LBFGS([synthesis], lr=1, max_iter=max_iter)

        def step_func():
            global iter
            optimizer.zero_grad()
            loss = net(synthesis)
            loss.backward(retain_graph = True)
            if (iter + 1) % 10 == 0:
                print('res-%d-iteration-%d: %f' % (i+1, iter + 1, loss.item()))
            # save image
            if (iter + 1) % sample_step == 0 or iter + 1 == max_iter:
                img = denormalizer(synthesis.clone().squeeze().to(torch.device("cpu"))).unsqueeze(0)
                img = F.interpolate(img, size=content.shape[2:4], mode="bilinear")
                torchvision.utils.save_image(img.squeeze(), 'res-%d-result-%d.jpg' % (i+1, iter + 1))

            iter += 1
            if iter == max_iter:
                iter = 0
            return loss
        optimizer.step(step_func)

if __name__ == "__main__":
    main()
