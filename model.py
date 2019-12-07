from __future__ import print_function
import torch.nn as nn
import torchvision.models as models
from loss_libs import ContentLoss, MRFStyleLoss, TVLoss

class MRFCNN(nn.Module):
    def __init__(self, content_img, style_img, content_loss_weight, tvloss_weight, patch_size, device):
        super(MRFCNN, self).__init__()
        self.content_loss_weight = content_loss_weight
        self.tvloss_weight = tvloss_weight
        self.MRFStyleLoss_idx = [11, 20]
        self.ContentLoss_idx = [22]
        self.patch_size = patch_size
        self.device = device

        self.ContentLosses = []
        self.MRFStyleLosses = []
        self.TVLoss = []

        self.target_content_feature_maps = []
        self.target_style_feature_maps = []

        self.content_hook_layers = []
        self.style_hook_layers = []
        self.tvloss_hook = None

        self.model = construct_model(content_img, style_img)

    def update_content_and_style_img(self, new_content_img, new_style_img):
        self.model(new_content_img.clone())
        for i, each in enumerate(content_hook_layers):
            self.target_content_feature_maps[i] = each.get_feature_map()

        self.model(new_style_img.clone())
        for i, each in enumerate(style_hook_layers):
            self.target_style_feature_maps[i] = each.get_feature_map()

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
        content_ret = []
        for each in content_hook_layers:
            content_ret.append(each.get_feature_map())

        style_ret = []
        for each in style_hook_layers:
            style_ret.append(each.get_feature_map())

        return tvloss_hook.get_feature_map(), content_ret, style_ret

    def calculate_total_loss(self, tv_feature, content_features, style_features):
        tv_loss = tv_loss_fn(input)

        content_loss = 0.0
        for i, each in enumerate(content_features):
            content_loss += content_loss_fn(each, self.target_content_feature_maps[i])

        style_loss = 0.0
        for i, each in enumerate(style_feature):
            style_loss += mrf_style_loss_fn(each, self.target_style_feature_maps[i]);

        total_loss = self.tvloss_weight * tv_loss + style_loss + self.content_loss_weight * content_loss

        return total_loss

    def construct_model(self, content_img, style_img):
        # create modified VGG for feature map inversion
        vgg19 = models.vgg19(pretrained=True)
        model = nn.Sequential()
        tv_hook = FeatureMapHook(True, self.device)
        model.add_module('TVLoss_hook_1', hook).to(self.device);
        mrf_cnt = 1
        content_cnt = 1

        for i in range(len(vgg19.features)):
            if (content_cnt - 1) == len(ContentLoss_idx) and (mrf_cnt - 1) == len(MRFStyleLoss_idx):
                break
            model.add_module('vgg_{}'.format(i), vgg19[i]).to(self.device);

            # MRFs layer after relu3_1 and relu4_1, so 11th and 20th
            if i in MRFStyleLoss_idx:
                target_style_feature_maps.append(model(style_img).detach())
                hook = FeatureMapHook(True, self.device)
                model.add_module('MRFStyleLoss_hook_{}'.format(mrf_cnt), hook).to(self.device)
                style_hook_layers.append(hook)
                mrf_cnt += 1

            # ContentLoss layer after relu4_2, so 22th
            if i in ContentLoss_idx:
                target_content_feature_maps.append(model(content_img).detach())
                hook = FeatureMapHook(True, self.device)
                model.add_module('ContentLoss_hook_{}'.format(content_cnt), hook).to(self.device)
                content_hook_layers.append(hook)
                content_cnt += 1

        return model
