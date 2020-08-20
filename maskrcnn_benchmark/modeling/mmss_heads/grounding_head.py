import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.utils.logged_module import LoggedModule

def wrap_nd_batch(fn, tensor):
    s = list(tensor.shape)
    tensor = tensor.reshape([-1, s[-1]])
    result = fn(tensor)
    s2 = list(result.shape)
    assert np.prod(s[:-1]) == s2[0]
    result = result.reshape(s[:-1] + s2[1:])
    return result

def choose_one(tensor):
    return wrap_nd_batch(
        lambda x: torch.multinomial(x, num_samples=1).squeeze(-1),
        tensor
    )

def remove_diag(square_matrix, dim):
    '''
    Removes the diagonal from a given matrix.
    Input is an NxN torch.Tensor.
    Returns an Nx(N-1) or (N-1)xN tensor depending on dim (1 or 0 respectively)
    '''
    assert(len(square_matrix.shape) == 2 and
           square_matrix.shape[0] == square_matrix.shape[1] and
           dim in [0, 1])
    N = square_matrix.shape[0]
    mask = (1-torch.eye(N)).to(torch.bool).to('cuda')
    if dim == 1:
        return torch.masked_select(square_matrix, mask).reshape([N, N - 1])
    if dim == 0:
        return torch.masked_select(square_matrix.t(), mask).reshape([N, N - 1]).t()


class GroundingHead(LoggedModule):
    def __init__(self, config, v_dim, l_dim):
        super(GroundingHead, self).__init__()
        self.config = config.MODEL.MMSS_HEAD.GROUNDING
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.v2l_projection = nn.Linear(self.v_dim, self.l_dim)

        # local similarity/distance metric can be either 'dot' or 'cosine' or 'euclidean'
        self.local_metric = self.config.LOCAL_METRIC

        # global distance metric can be either reconstruction_mse or aligned_local
        self.global_metric = self.config.GLOBAL_METRIC

        # word to region alignment method can be either 
        # 'softmax' or 'hardmax' or 'random_top3' or random_categorical
        self.alignment = self.config.ALIGNMENT
        
        # temperature is used as a denominator for the exponent of softmax, to make it smoother
        self.temperature = self.config.ALIGNMENT_TEMPERATURE

        # loss type can either be 'matching' or 'cross_entropy' or 'triplet'
        self.loss_type = self.config.LOSS

        # for triplet loss, negative mining method can be 'hardest', 'easiest', or 'random'
        self.negative_mining = self.config.NEGATIVE_MINING

        # distance margin for triplet loss
        self.margin = self.config.TRIPLET_MARGIN

        # whether to align each visual region to all caption words, or vice versa, or both
        self.align_words = self.config.ALIGN_WORDS_TO_REGIONS
        self.align_regions = self.config.ALIGN_REGIONS_TO_WORDS
        assert(self.align_words or self.align_regions)


    def forward(self, input_image, input_caption):
        caption_emb = input_caption['input_embeddings']
        caption_mask = input_caption['attention_mask'] * (1 - input_caption['special_tokens_mask'])
        self.log('attention_mask', input_caption['attention_mask'])
        self.log('special_tokens_mask', input_caption['special_tokens_mask'])
        self.log('caption_mask', caption_mask)
        self.log('caption_emb', caption_emb)
        caption_mask = caption_mask.to(torch.float32)
        num_words = caption_mask.sum(dim=1)
        _, max_num_words = caption_mask.shape

        region_features = input_image['region_features']
        region_mask = input_image['region_mask']
        region_mask = region_mask.to(torch.float32)
        num_regions = region_mask.sum(dim=1)
        batch_size, max_num_regions, _ = region_features.shape

        image_emb = self.v2l_projection(region_features).permute(0, 2, 1)

        if self.loss_type == 'cross_entropy' or self.loss_type == 'triplet':
            # we should compute the image-sentence distances for all image-sentence pairs 
            # in the batch, rather than only matching ones. So we replicate them BxB times.
            image_emb = image_emb[None, :, :, :].repeat(batch_size, 1, 1, 1).reshape(
                batch_size**2, self.l_dim, max_num_regions)
            caption_emb = caption_emb[:, None, :, :].repeat(1, batch_size, 1, 1).reshape(
                batch_size**2, max_num_words, self.l_dim)
            region_mask = region_mask[None, :, :].repeat(batch_size, 1, 1).reshape(
                batch_size**2, max_num_regions)
            caption_mask = caption_mask[:, None, :].repeat(1, batch_size, 1).reshape(
                batch_size**2, max_num_words)
            num_regions = num_regions[None, :].repeat(batch_size, 1).reshape(
                batch_size**2)
            num_words = num_words[:, None].repeat(1, batch_size).reshape(
                batch_size**2)

        if self.local_metric == 'dot':
            local_similarity = torch.bmm(caption_emb, image_emb)
            local_distance = - local_similarity
        elif self.local_metric == 'cosine':
            local_similarity = torch.bmm(caption_emb, image_emb)
            i_norm = (image_emb ** 2).sum(dim=1, keepdim=True).sqrt()
            c_norm = (caption_emb ** 2).sum(dim=2, keepdim=True).sqrt()
            local_similarity = local_similarity / (i_norm * c_norm)
            local_similarity = torch.where(
                torch.isnan(local_similarity), 
                torch.zeros_like(local_similarity), 
                local_similarity)
            local_distance = 1 - local_similarity
        elif self.local_metric == 'euclidean':
            local_similarity = torch.bmm(caption_emb, image_emb)
            i_norm = (image_emb ** 2).sum(dim=1, keepdim=True)
            c_norm = (caption_emb ** 2).sum(dim=2, keepdim=True)
            local_distance = i_norm + c_norm - (2 * local_similarity)
            # This implementation takes too much memory:
            # local_distance = ((image_emb[:, None, :, :] - caption_emb[:, :, :, None]) ** 2).sum(dim=2)
            local_similarity = - local_distance
        else:
            raise NotImplementedError

        local_similarity = local_similarity / self.temperature
        local_distance = local_distance / self.temperature

        self.log('local_similarity', local_similarity)
        local_similarity = torch.where(
            (caption_mask[:, :, None] * region_mask[:, None, :]) > 0,
            local_similarity, 
            local_similarity.min().detach() - 100.0
        )

        if self.alignment == 'softmax':
            if self.align_words:
                attention_w2r = F.softmax(local_similarity, dim=2)
            if self.align_regions:
                attention_r2w = F.softmax(local_similarity, dim=1)
        elif self.alignment == 'hardmax':
            if self.align_words:
                idx = torch.argmax(local_similarity, dim=2)
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32)
            if self.align_regions:
                idx = torch.argmax(local_similarity, dim=1)
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).permute(0, 2, 1)
        elif self.alignment == 'random_categorical':
            if self.align_words:
                attention_w2r = F.softmax(local_similarity, dim=2)
                idx = choose_one(attention_w2r)
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32)
            if self.align_regions:
                attention_r2w = F.softmax(local_similarity, dim=1).permute(0, 2, 1)
                idx = choose_one(attention_r2w)
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).permute(0, 2, 1)
        elif self.alignment == 'random_top3':
            if self.align_words:
                idx = torch.topk(local_similarity, k=3, dim=2).indices
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32).sum(dim=2)
                idx = choose_one(attention_w2r)
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32)
            if self.align_regions:
                idx = torch.topk(local_similarity, k=3, dim=1).indices
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).sum(dim=1)
                idx = choose_one(attention_r2w)
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).permute(0, 2, 1)
        elif self.alignment == 'optimal_transport':
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.align_words:
            self.log('attention_w2r', attention_w2r)
        if self.align_regions:
            self.log('attention_r2w', attention_r2w)

        if self.global_metric == 'reconstruction_mse':
            if self.align_words:
                caption_rec = torch.bmm(attention_w2r, image_emb.transpose(1, 2))
                global_dist_w2r = ((caption_rec - caption_emb) ** 2).mean(dim=2)
                global_dist_w2r = (
                    (global_dist_w2r * caption_mask).sum(dim=1) /
                    torch.max(num_words, other=torch.ones_like(num_words))
                )
            if self.align_regions:
                image_rec = torch.bmm(caption_emb.transpose(1, 2), attention_r2w)
                global_dist_r2w = ((image_rec - image_emb) ** 2).mean(dim=2).mean(dim=1)
                global_dist_r2w = (
                    (global_dist_r2w * region_mask).sum(dim=1) /
                    torch.max(num_regions, other=torch.ones_like(num_regions))
                )

        elif self.global_metric == 'aligned_local':
            if self.align_words:
                attention_w2r = attention_w2r * caption_mask[:, :, None]
                global_dist_w2r = (
                    (attention_w2r * local_distance).sum(dim=2).sum(dim=1) /
                    torch.max(num_words, other=torch.ones_like(num_words))
                )
            if self.align_regions:
                attention_r2w = attention_r2w * region_mask[:, None, :]
                global_dist_r2w = (
                    (attention_r2w * local_distance).sum(dim=2).sum(dim=1) /
                    torch.max(num_regions, other=torch.ones_like(num_regions))
                )
        else:
            raise NotImplementedError

        if self.align_words:
            global_dist_w2r = torch.where(
                (num_words > 0) + (num_regions > 0),
                global_dist_w2r,
                global_dist_w2r.max().detach() + 100.0
            )
        if self.align_regions:
            global_dist_r2w = torch.where(
                (num_regions > 0) + (num_words > 0),
                global_dist_r2w,
                global_dist_r2w.max().detach() + 100.0
            )

        if self.align_words:
            self.log('global_dist_w2r', global_dist_w2r)
        if self.align_regions:
            self.log('global_dist_r2w', global_dist_r2w)

        losses = {}
        if self.loss_type == 'matching':
            if self.local_metric == 'dot':
                raise Exception('Matching loss is not defined for dot product\
                                 because dot product is unbounded')
            if self.align_words:
                losses['Image-Caption Matching Loss (Align Words)'] = global_dist_w2r.mean()
            if self.align_regions:
                losses['Image-Caption Matching Loss (Align Regions)'] = global_dist_r2w.mean()

        elif self.loss_type == 'cross_entropy':
            if self.align_words:
                pw_cost_w2r = global_dist_w2r.reshape(batch_size, batch_size)
                pw_logits_c_cap_w2r = torch.log_softmax(- pw_cost_w2r, dim=0)
                pw_logits_c_img_w2r = torch.log_softmax(- pw_cost_w2r, dim=1)
                losses['Cross-Entropy Loss (Align Words, Choose Caption)'] = (
                    torch.diag(- pw_logits_c_cap_w2r).mean())
                losses['Cross-Entropy Loss (Align Words, Choose Image)'] = (
                    torch.diag(- pw_logits_c_img_w2r).mean())
            if self.align_regions:
                pw_cost_r2w = global_dist_r2w.reshape(batch_size, batch_size)
                pw_logits_c_cap_r2w = torch.log_softmax(- pw_cost_r2w, dim=0)
                pw_logits_c_img_r2w = torch.log_softmax(- pw_cost_r2w, dim=1)
                losses['Cross-Entropy Loss (Align Regions, Choose Caption)'] = (
                    torch.diag(- pw_logits_c_cap_r2w).mean())
                losses['Cross-Entropy Loss (Align Regions, Choose Image)'] = (
                    torch.diag(- pw_logits_c_img_r2w).mean())

        elif self.loss_type == 'triplet':
            if self.align_words:
                pw_cost_w2r = global_dist_w2r.reshape(batch_size, batch_size)
                positive_dist_w2r = torch.diag(pw_cost_w2r)
                negative_cap_all_w2r = remove_diag(pw_cost_w2r, dim=0)
                negative_img_all_w2r = remove_diag(pw_cost_w2r, dim=1)
                if batch_size < 2:
                    negative_cap_dist_w2r = positive_dist_w2r + self.margin
                    negative_img_dist_w2r = positive_dist_w2r + self.margin
                elif self.negative_mining == 'hardest':
                    negative_cap_dist_w2r = negative_cap_all_w2r.min(dim=0).values
                    negative_img_dist_w2r = negative_img_all_w2r.min(dim=1).values
                elif self.negative_mining == 'easiest':
                    negative_cap_dist_w2r = negative_cap_all_w2r.max(dim=0).values
                    negative_img_dist_w2r = negative_img_all_w2r.max(dim=1).values
                elif self.negative_mining == 'random':
                    negative_cap_dist_w2r = negative_cap_all_w2r.gather(
                        index=torch.randint(batch_size - 1, (1, batch_size)).to('cuda'),
                        dim=0)[0, :]
                    negative_img_dist_w2r = negative_img_all_w2r.gather(
                        index=torch.randint(batch_size - 1, (batch_size, 1)).to('cuda'),
                        dim=1)[:, 0]
                losses['Triplet Loss (Align Words, Choose Caption)'] = torch.mean(
                    F.relu(positive_dist_w2r - negative_cap_dist_w2r + self.margin))
                losses['Triplet Loss (Align Words, Choose Image)'] = torch.mean(
                    F.relu(positive_dist_w2r - negative_img_dist_w2r + self.margin))
            if self.align_regions:
                pw_cost_r2w = global_dist_r2w.reshape(batch_size, batch_size)
                positive_dist_r2w = torch.diag(pw_cost_r2w)
                negative_cap_all_r2w = remove_diag(pw_cost_r2w, dim=0)
                negative_img_all_r2w = remove_diag(pw_cost_r2w, dim=1)
                if batch_size < 2:
                    negative_cap_dist_r2w = positive_dist_r2w + self.margin
                    negative_img_dist_r2w = positive_dist_r2w + self.margin
                elif self.negative_mining == 'hardest':
                    negative_cap_dist_r2w = negative_cap_all_r2w.min(dim=0).values
                    negative_img_dist_r2w = negative_img_all_r2w.min(dim=1).values
                elif self.negative_mining == 'easiest':
                    negative_cap_dist_r2w = negative_cap_all_r2w.max(dim=0).values
                    negative_img_dist_r2w = negative_img_all_r2w.max(dim=1).values
                elif self.negative_mining == 'random':
                    negative_cap_dist_r2w = negative_cap_all_r2w.gather(
                        index=torch.randint(batch_size - 1, (1, batch_size)).to('cuda'),
                        dim=0)[0, :]
                    negative_img_dist_r2w = negative_img_all_r2w.gather(
                        index=torch.randint(batch_size - 1, (batch_size, 1)).to('cuda'),
                        dim=1)[:, 0]
                losses['Triplet Loss (Align Regions, Choose Caption)'] = torch.mean(
                    F.relu(positive_dist_r2w - negative_cap_dist_r2w + self.margin))
                losses['Triplet Loss (Align Regions, Choose Image)'] = torch.mean(
                    F.relu(positive_dist_r2w - negative_img_dist_r2w + self.margin))
        else:
            raise NotImplementedError

        other_info = {}
        if self.loss_type == 'cross_entropy' or self.loss_type == 'triplet':
            if self.align_words:
                other_info['Batch Accuracy (Align Words, Choose Caption)'] = torch.mean(
                    (pw_cost_w2r.argmin(dim=0) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))
                other_info['Batch Accuracy (Align Words, Choose Image)'] = torch.mean(
                    (pw_cost_w2r.argmin(dim=1) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))
            if self.align_regions:
                other_info['Batch Accuracy (Align Regions, Choose Caption)'] = torch.mean(
                    (pw_cost_r2w.argmin(dim=0) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))
                other_info['Batch Accuracy (Align Regions, Choose Image)'] = torch.mean(
                    (pw_cost_r2w.argmin(dim=1) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))

        self.log_dict(losses)
        self.log_dict(other_info)

        return other_info, losses
