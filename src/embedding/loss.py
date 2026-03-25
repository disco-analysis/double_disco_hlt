import torch
import torch.nn as nn
import torch.nn.functional as F


def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    var_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1), len(var_1))
    yy = var_1.repeat(len(var_1), 1).view(len(var_1), len(var_1))
    amat = (xx - yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2), len(var_2))
    yy = var_2.repeat(len(var_2), 1).view(len(var_2), len(var_2))
    bmat = (xx - yy).abs()

    amatavg = torch.mean(amat * normedweight, dim=1)
    Amat = amat - amatavg.repeat(len(var_1), 1).view(len(var_1), len(var_1)) \
        - amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1), len(var_1)) \
        + torch.mean(amatavg * normedweight)

    bmatavg = torch.mean(bmat * normedweight, dim=1)
    Bmat = bmat - bmatavg.repeat(len(var_2), 1).view(len(var_2), len(var_2)) \
        - bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2), len(var_2)) \
        + torch.mean(bmatavg * normedweight)

    ABavg = torch.mean(Amat * Bmat * normedweight, dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight, dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight, dim=1)

    if power == 1:
        dCorr = (torch.mean(ABavg * normedweight)) / torch.sqrt(
            (torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)).clamp(min=1e-8)
        )
    elif power == 2:
        dCorr = (torch.mean(ABavg * normedweight)) ** 2 / (
            torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)
        ).clamp(min=1e-8)
    else:
        dCorr = ((torch.mean(ABavg * normedweight)) / torch.sqrt(
            (torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)).clamp(min=1e-8)
        )) ** power

    return dCorr

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, class_similarity=None):
        super().__init__()
        self.temperature = temperature
        self.class_similarity = class_similarity

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)

        positive_mask = torch.eq(labels, labels.T).float()
        positive_mask.fill_diagonal_(0)

        negatives_mask = (~torch.eye(batch_size, dtype=torch.bool, device=device)).float()
     
        neg_weights = negatives_mask

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix) * neg_weights
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = sim_matrix - log_sum_exp
        pos_log_prob = log_prob * positive_mask

        num_positives = positive_mask.sum(dim=1)
        valid_samples = num_positives > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sample_losses = pos_log_prob.sum(dim=1) / (num_positives + 1e-8)
        sample_losses = sample_losses[valid_samples]

        return -sample_losses.mean()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Source: https://github.com/HobbitLong/SupContrast
    """
    def __init__(
        self, 
        temperature = 0.07, 
        contrast_mode = 'all',
        base_temperature = 0.07
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            # raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            #  'at least 3 dimensions are required')
            features = features.unsqueeze(1)
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-8))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss