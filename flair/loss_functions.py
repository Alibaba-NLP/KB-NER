import torch.nn.functional as F
from torch import nn, Tensor
import torch
import pdb

def pytorch_cos_sim(a,b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))



def multiple_negatives_ranking_loss(embeddings_a, embeddings_b, temperature = 20.0):
    loss_model = nn.CrossEntropyLoss()
    scores = pytorch_cos_sim(embeddings_a, embeddings_b) * temperature
    labels = torch.tensor(range(len(scores)), dtype=torch.long,
                          device=scores.device)  # Example a[i] should match with b[i]
    return loss_model(scores, labels), (scores.argmax(-1) == labels).sum()


def multiple_negatives_ranking_loss_with_labels(embeddings_a, embeddings_b, labels, temperature = 20.0):
    # labels: batch * 1
    # loss_model = nn.CrossEntropyLoss()
    scores = pytorch_cos_sim(embeddings_a, embeddings_b)
    target_labels = torch.zeros_like(scores)
    target_labels[range(len(scores)),range(len(scores))] = labels
    # labels = torch.tensor(range(len(scores)), dtype=torch.long,
    #                       device=scores.device)  # Example a[i] should match with b[i]
    # - (torch.log(torch.sigmoid(scores))*target_labels + (torch.log(1-torch.sigmoid(scores))*(1-target_labels))).sum()
    
    return F.binary_cross_entropy_with_logits(scores, target_labels), (scores[range(len(scores)),range(len(scores))].sigmoid())


def simclr(embeddings_a, embeddings_b, temperature=20.0):
    all_embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)
    similarities = pytorch_cos_sim(all_embeddings, all_embeddings) * temperature

    batch_size = list(embeddings_a.size())[0]
    similarities += torch.diag(torch.ones(batch_size * 2)).to(similarities.device) * -1e30
    labels = torch.cat(
        [
            torch.range(0, batch_size-1, dtype=torch.long) + batch_size,
            torch.range(0, batch_size-1, dtype=torch.long)
        ], dim=0
    )
    loss = nn.CrossEntropyLoss(ignore_index=-1)(similarities, labels.to(similarities.device))
    return loss



def clip(embeddings_a, embeddings_b, temperature=20.0):
    scores = pytorch_cos_sim(embeddings_a, embeddings_b) * temperature
    _labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(scores, _labels)
    loss_ = cross_entropy_loss(scores.transpose(0,1), _labels)
    loss = (loss + loss_) / 2.
    return loss


def circle_loss(embeddings_a, embeddings_b, margin, gamma):

    bs = embeddings_a.size(0)
    norm_a = embeddings_a / embeddings_a.norm(dim=1)[:, None]
    norm_b = embeddings_b / embeddings_b.norm(dim=1)[:, None]
    neg_cos_all = torch.matmul(norm_a, norm_b.transpose(0, 1).contiguous())  # B * B

    pos_cosine = torch.diag(neg_cos_all).reshape(-1, 1)
    neg_cosine = torch.ones_like(neg_cos_all)
    neg_cosine[:, :bs] -= torch.eye(bs, device=neg_cos_all.device)
    neg_mask = neg_cosine.bool().to(neg_cos_all.device)
    neg_cosine = neg_cos_all.masked_select(neg_mask).reshape(bs, -1)
    pos_loss = torch.sum(torch.exp(-gamma * ((1. + margin - pos_cosine) * (pos_cosine - 1. + margin))), dim=1)
    neg_loss = torch.sum(torch.exp(gamma * ((neg_cosine + margin) * (neg_cosine - margin))), dim=1)
    circle_loss = torch.mean(torch.log(1. + neg_loss * pos_loss))
    return circle_loss