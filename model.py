from __future__ import print_function
import torch.nn as nn
import torchvision.models as models
from loss_libs import ContentLoss, MRFStyleLoss, TVLoss, FeatureMapHook

class MRFCNN(nn.Module):
    def __init__(self, content_img, style_img, style_loss_weight, tvloss_weight, style_stride, syn_stride, gpu_chunk_size,  device):
        super(MRFCNN, self).__init__()
        self.style_loss_weight = style_loss_weight
        self.tvloss_weight = tvloss_weight
        self.MRFStyleLoss_idx = [11, 20]
        self.ContentLoss_idx = [22]
        self.patch_size = 3
        self.device = device

        self.style_stride = style_stride
        self.syn_stride = syn_stride
        self.gpu_chunk_size = gpu_chunk_size

        self.ContentLosses = []
        self.MRFStyleLosses = []
        self.tv_loss_fn = None

        self.target_content_feature_maps = []
        self.target_style_feature_maps = []

        self.content_hook_layers = []
        self.style_hook_layers = []
        self.tvloss_hook = None

        self.model = self.construct_model(content_img, style_img)
        self.update_content_and_style_img(content_img, style_img)

    def update_content_and_style_img(self, new_content_img, new_style_img):
        print("------updating content and style images-------")
        self.model(new_content_img.clone())
        for i, each in enumerate(self.content_hook_layers):
            self.target_content_feature_maps[i] = each.get_feature_map()
            self.ContentLosses[i].update(self.target_content_feature_maps[i])

        self.model(new_style_img.clone())
        for i, each in enumerate(self.style_hook_layers):
            self.target_style_feature_maps[i] = each.get_feature_map()
            self.MRFStyleLosses[i].update(self.target_style_feature_maps[i])

    def forward(self, synth_img):
        """
        returns:
            tvloss: feature map from tvloss hook
            content_ret: list of feature maps from respective content hook
            style_ret: list of feature maps from respective style hook
        """
        self.model(synth_img)

        # for each_style_feature_maps in range(target_style_feature_maps):
        #     total_loss += MRFStyleLoss()

        tv_loss = self.tv_loss_fn(self.tvloss_hook.get_feature_map())

        content_loss = 0.0
        for i, each in enumerate(self.content_hook_layers):
            content_loss += self.ContentLosses[i](each.get_feature_map())

        style_loss = 0.0
        for i, each in enumerate(self.style_hook_layers):
            style_loss += self.MRFStyleLosses[i](each.get_feature_map());

        total_loss = self.tvloss_weight * tv_loss + self.style_loss_weight * style_loss + content_loss

        return total_loss.requires_grad_(True)

    def construct_model(self, content_img, style_img):
        # create modified VGG for feature map inversion
        vgg19 = models.vgg19(pretrained=True).to(self.device)
        model = nn.Sequential()
        self.tvloss_hook = FeatureMapHook(False, self.device)
        model.add_module('TVLoss_hook_1', self.tvloss_hook)
        self.tv_loss_fn = TVLoss()
        mrf_cnt = 1
        content_cnt = 1

        for i in range(len(vgg19.features)):
            if (content_cnt - 1) == len(self.ContentLoss_idx) and (mrf_cnt - 1) == len(self.MRFStyleLoss_idx):
                break
            model.add_module('vgg_{}'.format(i), vgg19.features[i])

            # MRFs layer after relu3_1 and relu4_1, so 11th and 20th
            if i in self.MRFStyleLoss_idx:
                feature_map = model(style_img).detach()
                self.target_style_feature_maps.append(feature_map)
                self.MRFStyleLosses.append(MRFStyleLoss(feature_map, self.patch_size, self.device, self.gpu_chunk_size, self.syn_stride))
                hook = FeatureMapHook(False, self.device)
                model.add_module('MRFStyleLoss_hook_{}'.format(mrf_cnt), hook)
                self.style_hook_layers.append(hook)
                mrf_cnt += 1

            # ContentLoss layer after relu4_2, so 22th
            if i in self.ContentLoss_idx:
                feature_map = model(content_img).detach()
                self.target_content_feature_maps.append(feature_map)
                self.ContentLosses.append(ContentLoss(self.device))
                hook = FeatureMapHook(False, self.device)
                model.add_module('ContentLoss_hook_{}'.format(content_cnt), hook)
                self.content_hook_layers.append(hook)
                content_cnt += 1

        return model
