import torch
from torch.nn import functional as F

class WSDDNLossComputation(object):
    """
    Computes the loss for WSDDN, which is a multi-label image-level binary cross-entropy loss
    """
    def __init__(self, cfg):
        self.config = cfg
        self.background_weight = cfg.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT_BACKGROUND


    def __call__(self, class_logits, targets, num_box_per_img):
        """
        Arguments:
            class_logits (Tensor)
            targets (Tensor): image-level multi-label target. Each row is a binary vector of lenth num_classes.
            num_box_per_img (list[int])

        Returns:
            classification_loss (Tensor)
        """
        device = class_logits.device
        box_class_logits = class_logits.split(num_box_per_img, dim=0)
        image_class_logits = [torch.logsumexp(l, dim=0) for l in box_class_logits]
        image_class_logits = torch.stack(image_class_logits, dim=0)
        negative_logits = torch.log(1.0 - torch.exp(image_class_logits) + 1e-6)
        classification_loss = (- (targets * image_class_logits) - 
            ((1 - targets) * negative_logits * self.background_weight))
        classification_loss = classification_loss.mean()
        return classification_loss


def make_roi_box_loss_evaluator(cfg):
    loss_evaluator = WSDDNLossComputation(cfg)
    return loss_evaluator
