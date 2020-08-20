from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from . import transformers
from . import word_embedding


@registry.LANGUAGE_BACKBONES.register("BERT-Base")
def build_bert_backbone(cfg):
    body = transformers.BERT(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model

@registry.LANGUAGE_BACKBONES.register("WordEmbedding")
def build_embedding_backbone(cfg):
    body = word_embedding.WordEmbedding(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model

def build_backbone(cfg):
    assert cfg.MODEL.LANGUAGE_BACKBONE.TYPE in registry.LANGUAGE_BACKBONES, \
        "cfg.MODEL.LANGUAGE_BACKBONE.TYPE: {} is not registered in registry".format(
            cfg.MODEL.LANGUAGE_BACKBONE.TYPE
        )
    return registry.LANGUAGE_BACKBONES[cfg.MODEL.LANGUAGE_BACKBONE.TYPE](cfg)
