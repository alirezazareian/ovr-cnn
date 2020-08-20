from copy import deepcopy
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.update_bert_config()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained(
            'bert-base-uncased', config=self.bert_config)    
        self.freeze()
        self.out_channels = self.bert_config.hidden_size
        head_config = self.config.MODEL.MMSS_HEAD.TRANSFORMER
        self.mlm = head_config.MASKED_LANGUAGE_MODELING
        self.mlm_prob = head_config.MASKED_LANGUAGE_MODELING_PROB
        self.mlm_prob_mask = head_config.MASKED_LANGUAGE_MODELING_PROB_MASK
        self.mlm_prob_noise = head_config.MASKED_LANGUAGE_MODELING_PROB_NOISE
        self.mlm_during_validation = head_config.MASKED_LANGUAGE_MODELING_VALIDATION
        self.embeddings = self.bert_model.embeddings.word_embeddings.weight

    def forward(self, text_list):
        tokenized_batch = self.tokenizer.batch_encode_plus(text_list, 
            add_special_tokens=True, 
            pad_to_max_length=True,
            return_special_tokens_mask=True,
        )
        if self.mlm:
            tokenized_batch['target_ids'] = deepcopy(tokenized_batch['input_ids'])
            tokenized_batch['mlm_mask'] = []
            for i, item in enumerate(tokenized_batch['input_ids']):
                mlm_mask = []
                for j in range(len(item)):
                    if (tokenized_batch['special_tokens_mask'][i][j] or
                        not tokenized_batch['attention_mask'][i][j] or
                        not (self.training or self.mlm_during_validation)):
                        mlm_mask.append(0)
                        continue
                    prob = np.random.rand()
                    if prob < self.mlm_prob:
                        mlm_mask.append(1)
                        prob /= self.mlm_prob
                        if prob < self.mlm_prob_mask:
                            item[j] = self.tokenizer.convert_tokens_to_ids(
                                self.tokenizer.mask_token)
                            tokenized_batch['special_tokens_mask'][i][j] = 1
                        elif prob < self.mlm_prob_mask + self.mlm_prob_noise:
                            item[j] = np.random.randint(len(self.tokenizer))
                    else:
                        mlm_mask.append(0)
                tokenized_batch['mlm_mask'].append(mlm_mask)

        tokenized_batch = {k: torch.tensor(v).cuda() for k, v in tokenized_batch.items()}
        bert_output = self.bert_model(
            input_ids=tokenized_batch['input_ids'],
            attention_mask=tokenized_batch['attention_mask'],
        )
        tokenized_batch['encoded_tokens'] = bert_output[0]

        tokenized_batch['input_embeddings'] = self.embeddings[tokenized_batch['input_ids']]
        return tokenized_batch


    def freeze(self):
        for p in self.bert_model.pooler.parameters():
            p.requires_grad = False
        if self.config.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.parameters():
                p.requires_grad = False


    def update_bert_config(self):
        pass