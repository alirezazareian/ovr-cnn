from copy import deepcopy
import numpy as np
import torch
from torch import nn
from transformers.tokenization_bert import BasicTokenizer

class WordEmbedding(nn.Module):
    def __init__(self, config):
        super(WordEmbedding, self).__init__()
        self.config = config
        self.tokenizer = BasicTokenizer(do_lower_case=True)
        # standard deviation of initialization
        init_std = config.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.initializer_range

        self.words = []
        self.word2idx = {}
        self.embeddings = []
        with open(config.MODEL.LANGUAGE_BACKBONE.EMBEDDING_PATH, 'r') as fin:
            for row in fin:
                row_tk = row.split()
                self.words.append(row_tk[0])
                self.word2idx[row_tk[0]] = len(self.words) - 1
                self.embeddings.append([float(num) for num in row_tk[1:]])

        self.embeddings = torch.tensor(
            np.asarray(self.embeddings, dtype=np.float32)).cuda()
        self.embeddings = nn.Parameter(self.embeddings)
        self.out_channels = self.embeddings.shape[-1]
        if self.config.MODEL.LANGUAGE_BACKBONE.FREEZE:
            self.embeddings.requires_grad = False

        self.words.extend(['[OOV]', '[PAD]', '[CLS]', '[SEP]', '[MASK]'])
        self.oov_idx = len(self.words) - 5
        self.pad_idx = len(self.words) - 4
        self.cls_idx = len(self.words) - 3
        self.sep_idx = len(self.words) - 2
        self.mask_idx = len(self.words) - 1
        self.special_tokens = set([self.oov_idx, self.pad_idx, self.cls_idx,
                                   self.sep_idx, self.mask_idx])
        self.special_embeddings = nn.Parameter(torch.zeros(5, self.out_channels).cuda())
        self.special_embeddings.data.normal_(mean=0.0, std=init_std)
        self.aug_embeddings = torch.cat([self.embeddings, self.special_embeddings], dim=0)
        head_config = self.config.MODEL.MMSS_HEAD.TRANSFORMER
        self.mlm = head_config.MASKED_LANGUAGE_MODELING
        self.mlm_prob = head_config.MASKED_LANGUAGE_MODELING_PROB
        self.mlm_prob_mask = head_config.MASKED_LANGUAGE_MODELING_PROB_MASK
        self.mlm_prob_noise = head_config.MASKED_LANGUAGE_MODELING_PROB_NOISE
        self.mlm_during_validation = head_config.MASKED_LANGUAGE_MODELING_VALIDATION
        self.add_position_embedding = config.MODEL.LANGUAGE_BACKBONE.ADD_POSITION_EMBEDDING
        if self.add_position_embedding:
            # maximum length of a sentence
            m = config.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.max_position_embeddings
            self.position_embedding = nn.Parameter(torch.zeros(m, self.out_channels))
            self.position_embedding.data.normal_(mean=0.0, std=init_std)

    def forward(self, text_list):
        tokenized_batch = {
            'input_ids': [],
            'attention_mask': [],
            'encoded_tokens': [],
            'input_embeddings': [],
            'special_tokens_mask': [],
        }
        for i in range(len(text_list)):
            tokens = self.tokenizer.tokenize(text_list[i])
            ids = [self.word2idx.get(t, self.oov_idx) for t in tokens]
            ids = [self.cls_idx] + ids + [self.sep_idx]
            tokenized_batch['input_ids'].append(ids)

        max_len = max([len(i) for i in tokenized_batch['input_ids']])
        for i in range(len(text_list)):
            ids = tokenized_batch['input_ids'][i]
            l = len(ids)
            ids.extend([self.pad_idx] * (max_len - l))

        if self.mlm:
            tokenized_batch['target_ids'] = deepcopy(tokenized_batch['input_ids'])
            tokenized_batch['mlm_mask'] = []
            for i, item in enumerate(tokenized_batch['input_ids']):
                mlm_mask = []
                for j in range(len(item)):
                    if (item[j] in self.special_tokens or
                        not (self.training or self.mlm_during_validation)):
                        mlm_mask.append(0)
                        continue
                    prob = np.random.rand()
                    if prob < self.mlm_prob:
                        mlm_mask.append(1)
                        prob /= self.mlm_prob
                        if prob < self.mlm_prob_mask:
                            item[j] = self.mask_idx
                        elif prob < self.mlm_prob_mask + self.mlm_prob_noise:
                            # assuming special tokens are at the end of the words list
                            item[j] = np.random.randint(
                                len(self.words) - len(self.special_tokens))
                    else:
                        mlm_mask.append(0)
                tokenized_batch['mlm_mask'].append(mlm_mask)

        for i in range(len(text_list)):
            ids = np.asarray(tokenized_batch['input_ids'][i])
            tokenized_batch['attention_mask'].append(
                (ids != self.pad_idx).astype(np.int64))
            enc = self.aug_embeddings[ids]
            tokenized_batch['input_embeddings'].append(enc)
            if self.add_position_embedding:
                enc = enc + self.position_embedding[:max_len]
            tokenized_batch['encoded_tokens'].append(enc)
            sp_mask = []
            for tk in ids:
                if tk in self.special_tokens:
                    sp_mask.append(1)
                else:
                    sp_mask.append(0)
            tokenized_batch['special_tokens_mask'].append(sp_mask)

        tokenized_batch['input_embeddings'] = torch.stack(
                tokenized_batch['input_embeddings'], dim=0)
        tokenized_batch['encoded_tokens'] = torch.stack(
                tokenized_batch['encoded_tokens'], dim=0)
        tokenized_batch['input_ids'] = torch.tensor(
            tokenized_batch['input_ids']).cuda()
        tokenized_batch['attention_mask'] = torch.tensor(
            tokenized_batch['attention_mask']).cuda()
        tokenized_batch['special_tokens_mask'] = torch.tensor(
            tokenized_batch['special_tokens_mask']).cuda()
        if self.mlm:
            tokenized_batch['mlm_mask'] = torch.tensor(
                tokenized_batch['mlm_mask']).cuda()
            tokenized_batch['target_ids'] = torch.tensor(
                tokenized_batch['target_ids']).cuda()
        return tokenized_batch
