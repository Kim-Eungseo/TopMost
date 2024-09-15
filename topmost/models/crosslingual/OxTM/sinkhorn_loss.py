import torch
import torch.nn as nn


class SinkhornLoss(nn.Module):
    def __init__(self, epsilon=0.1, num_iter=500):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.num_iter = num_iter

    def forward(self, a, b, M):
        # Normalize the distributions to sum to 1
        a = a / a.sum(dim=-1, keepdim=True)
        b = b / b.sum(dim=-1, keepdim=True)

        # Initialize u and v
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        # Kernel matrix K
        K = torch.exp(-M / self.epsilon)

        # Sinkhorn iterations
        for _ in range(self.num_iter):
            u = a / (K @ v)
            v = b / (K.transpose(0, 1) @ u)

        # Transport plan pi
        pi = torch.diag_embed(u) @ K @ torch.diag_embed(v)

        # Compute the Sinkhorn loss as the transport cost
        sinkhorn_loss = torch.sum(pi * M)

        return sinkhorn_loss
