import torch
from torch import nn

from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from maskrcnn_benchmark.utils.logged_module import LoggedModule


class WSDDNHead(LoggedModule):
    """
    Implementing Weakly Supervised Deep Detection Networks
    """

    def __init__(self, cfg, in_channels):
        super(WSDDNHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        if cfg.MODEL.ROI_BOX_HEAD.FREEZE_FEATURE_EXTRACTOR:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False


    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[ndarray], optional): the ground-truth captions.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): During testing, the predicted boxlists are returned. 
                                       During training, input proposals are bypassed.
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        self.log('features', x)
        # final classifier that converts the features into predictions
        num_box_per_img = [len(p) for p in proposals]
        class_logits = self.predictor(x, num_box_per_img)
        self.log('class_logits', class_logits)

        if not self.training:
            result = self.post_processor(class_logits, proposals)
            return x, result, {}

        targets = torch.tensor(targets).cuda()
        self.log('targets', targets)
        loss_classifier = self.loss_evaluator(class_logits, targets, num_box_per_img)
        self.log_dict({'loss_classifier': loss_classifier})
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier),
        )