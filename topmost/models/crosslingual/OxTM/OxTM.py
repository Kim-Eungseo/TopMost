import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from topmost.models.Encoder import MLPEncoder


class OxTM(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size_en: int,
                 vocab_size_cn: int,
                 pretrain_word_embeddings_en: torch.Tensor,
                 pretrain_word_embeddings_cn: torch.Tensor,
                 en1_units: int,
                 dropout: float = 0.0,
                 device_BWE='cuda'):
        super().__init__()

        self.num_topic = num_topics

        self.BWE_en = torch.from_numpy(pretrain_word_embeddings_en).float().to(device_BWE)
        self.BWE_cn = torch.from_numpy(pretrain_word_embeddings_cn).float().to(device_BWE)

        self.encoder_en = MLPEncoder(vocab_size_en, num_topics, en1_units, dropout)
        self.encoder_cn = MLPEncoder(vocab_size_cn, num_topics, en1_units, dropout)

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T),
                                requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (
                1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_en = nn.BatchNorm1d(vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_cn))))

        # 캐싱을 위한 변수들 초기화
        self.cached_sinkhorn_loss = None
        self.previous_beta_en_hash = None
        self.previous_beta_cn_hash = None

        # Define the SamplesLoss (Sinkhorn Loss) from GeomLoss
        self.sinkhorn_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05)

        # Normalize Bilingual Word Embeddings
        BWE_en_norm = F.normalize(self.BWE_en, p=2, dim=1)
        BWE_cn_norm = F.normalize(self.BWE_cn, p=2, dim=1)

        self.M = 1 - torch.mm(BWE_en_norm, BWE_cn_norm.T).to(device_BWE)  # Cost matrix = Cosine distance matrix

    def get_beta(self):
        beta_en = self.phi_en
        beta_cn = self.phi_cn
        return beta_en, beta_cn

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

    def forward(self, x_en, x_cn):
        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')

        beta_en, beta_cn = self.get_beta()
        rst_dict = dict()

        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')

        loss_en = self.loss_function(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.loss_function(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn
        rst_dict['loss_en'] = loss_en
        rst_dict['loss_cn'] = loss_cn

        reg = 1e-3

        sinkhorn_loss = self.sinkhorn_loss(beta_cn, beta_en, self.M, reg, self.num_topic)
        # sinkhorn_loss = self.sinkhorn_loss_fn(BWE_en_norm, BWE_cn_norm)

        rst_dict['sinkhorn_loss'] = sinkhorn_loss

        total_loss = loss + 0.5 * sinkhorn_loss  # weight of sinkhorn loss function
        rst_dict['loss'] = total_loss

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

    def sinkhorn_loss(self, beta_cn, beta_en, cost_matrix, reg, num_topic):
        #
        beta_cn = beta_cn.detach()
        beta_en = beta_en.detach()
        cost_matrix = cost_matrix.detach()

        total_topic_loss = 0
        top_n = 100

        for topic in range(num_topic):

            beta_cn_topic = beta_cn[topic]
            top_cn_indices = torch.topk(beta_cn_topic, top_n).indices  # Get top 10 indices directly with PyTorch

            beta_en_topic = beta_en[topic]
            top_en_indices = torch.topk(beta_en_topic, top_n).indices  # Get top 10 indices directly with PyTorch

            sub_beta_cn = beta_cn_topic[top_cn_indices]
            sub_beta_en = beta_en_topic[top_en_indices]

            sub_beta_cn /= sub_beta_cn.sum()
            sub_beta_en /= sub_beta_en.sum()

            sub_cost_matrix = cost_matrix[top_cn_indices][:, top_en_indices]

            sub_beta_cn_np = sub_beta_cn.cpu().numpy()
            sub_beta_en_np = sub_beta_en.cpu().numpy()
            sub_cost_matrix_np = sub_cost_matrix.cpu().numpy()

            # loss = ot.sinkhorn2(sub_beta_cn_np, sub_beta_en_np, sub_cost_matrix_np, reg=reg, numItermax=1000)
            loss = ot.sinkhorn2(sub_beta_cn_np, sub_beta_en_np, sub_cost_matrix_np, reg=reg, numItermax=1000)
            total_topic_loss += loss

        total_topic_loss = torch.tensor(float(total_topic_loss), device=beta_cn.device)  # Stay on GPU

        return total_topic_loss

