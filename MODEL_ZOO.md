## Model Zoo and Baselines

### Our pretrained backbones

backbone | type | initialization | model id
-- | -- | -- | -- 
R-50-C4 | Grounding + BERT | ImageNet | 121
R-50-C4 | Grounding + BERT | Random | 134


### Our object detectors

backbone | type | initialization | # seen | box AP 0.5 seen | box AP 0.5 unseen | model id
-- | -- | -- | -- | -- | -- | --
R-50-C4 | Zero-Shot | 121 | 48 | 47% | 27% | 130
R-50-C4 | Zero-Shot | 121 | 64 | 47% | ?? | 131
R-50-C4 | Zero-Shot | 134 | 64 | 47% | ?? | 135


### Original Faster and Mask R-CNN baselines from Facebook

backbone | type | lr sched | im / gpu | train mem(GB) | train time (s/iter) | total train time(hr) | inference time(s/im) | box AP | mask AP | model id
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
R-50-C4 | Fast | 1x | 1 | 5.8 | 0.4036 | 20.2 | 0.17130 | 34.8 | - | [6358800](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_C4_1x.pth)
