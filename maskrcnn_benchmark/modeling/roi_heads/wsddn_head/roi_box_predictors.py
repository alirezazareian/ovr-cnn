import torch
from torch import nn
import torch.nn.functional as F

class WSDDNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(WSDDNPredictor, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.cls_score = nn.Linear(in_channels, self.num_classes)
        self.det_score = nn.Linear(in_channels, self.num_classes)
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.det_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.det_score.bias, 0)
        self.embedding_based = False

    def forward(self, x, num_box_per_img):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)
        cls_logit = F.log_softmax(cls_logit, dim=1)
        det_logit = det_logit.split(num_box_per_img, dim=0)
        det_logit = [F.log_softmax(l, dim=0) for l in det_logit]
        det_logit = torch.cat(det_logit, dim=0)
        combined_logit = cls_logit + det_logit
        return combined_logit

def make_roi_box_predictor(cfg, in_channels):
    return WSDDNPredictor(cfg, in_channels)