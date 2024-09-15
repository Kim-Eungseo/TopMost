import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topmost.models.Encoder import MLPEncoder
from topmost.models.crosslingual.OxTM.sinkhorn_loss import SinkhornLoss


class OxTM(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size_anchor: int,
                 vocab_size_alignment: int,
                 pretrain_word_embeddings_anchor: torch.Tensor,
                 pretrain_word_embeddings_alignment: torch.Tensor,
                 anchor1_units: int,
                 dropout: float = 0.0,
                 device='cpu'):
        super().__init__()

        self.sinkhorn_loss_fn = None
        self.num_topic = num_topics

        self.BWE_anchor = torch.from_numpy(pretrain_word_embeddings_anchor).float()
        self.BWE_alignment = torch.from_numpy(pretrain_word_embeddings_alignment).float()

        self.BWE_anchor = self.BWE_anchor.to(device)
        self.BWE_alignment = self.BWE_alignment.to(device)

        self.encoder_anchor = MLPEncoder(vocab_size_anchor, num_topics, anchor1_units, dropout)
        self.encoder_alignment = MLPEncoder(vocab_size_alignment, num_topics, anchor1_units, dropout)

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T),
                                requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (
                1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_anchor = nn.BatchNorm1d(vocab_size_anchor, affine=True)
        self.decoder_bn_anchor.weight.requires_grad = False
        self.decoder_bn_alignment = nn.BatchNorm1d(vocab_size_alignment, affine=True)
        self.decoder_bn_alignment.weight.requires_grad = False

        self.phi_anchor = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_anchor))))
        self.phi_alignment = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_alignment))))

    def get_beta(self):
        beta_anchor = self.phi_anchor
        beta_alignment = self.phi_alignment
        return beta_anchor, beta_alignment

    def get_theta(self, x, lang):
        theta, mu, logvar = getattr(self, f'encoder_{lang}')(x)

        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def forward(self, x_anchor, x_alignment):
        theta_anchor, mu_anchor, logvar_anchor = self.get_theta(x_anchor, lang='anchor')
        theta_alignment, mu_alignment, logvar_alignment = self.get_theta(x_alignment, lang='alignment')

        beta_anchor, beta_alignment = self.get_beta()

        rst_dict = dict()

        x_recon_anchor = self.decode(theta_anchor, beta_anchor, lang='anchor')
        x_recon_alignment = self.decode(theta_alignment, beta_alignment, lang='alignment')

        loss_anchor = self.loss_function(x_recon_anchor, x_anchor, mu_anchor, logvar_anchor)
        loss_alignment = self.loss_function(x_recon_alignment, x_alignment, mu_alignment, logvar_alignment)

        loss = loss_anchor + loss_alignment
        rst_dict['loss_anchor'] = loss_anchor
        rst_dict['loss_alignment'] = loss_alignment

        # Normalize Bilingual Word Embeddings
        BWE_anchor_norm = F.normalize(self.BWE_anchor, p=2, dim=1)
        BWE_alignment_norm = F.normalize(self.BWE_alignment, p=2, dim=1)

        # Compute Cost matrix
        M = 1 - torch.mm(BWE_anchor_norm, BWE_alignment_norm.T)  # Cost matrix = Cosine distance matrix

        fea_anchor = beta_anchor.T
        fea_alignment = beta_alignment.T

        # Compute Sinkhorn loss between topic-word distributions
        # hyperparameter : entropy regularization, the number of sinkhorn algorithm iteration

        self.sinkhorn_loss_fn = SinkhornLoss(epsilon=0.1, num_iter=500)
        sinkhorn_loss = self.sinkhorn_loss_fn(fea_anchor, fea_alignment, M)
        rst_dict['sinkhorn_loss'] = sinkhorn_loss

        total_loss = loss + 0.1 * sinkhorn_loss  # weight of sinkorn loss function
        rst_dict['loss'] = total_loss

        # return total_loss, rst_dict
        return rst_dict

    def loss_function(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topic)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS
