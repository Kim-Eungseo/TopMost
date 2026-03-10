"""
EBM-LTM: Energy-Based Latent Topic Model

Unifying Energy-Based Priors and Amortized MCMC Inference
for Neural Topic Modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnergyNetwork(nn.Module):
    """
    2-layer MLP energy function with Swish (SiLU) activation and spectral
    normalization for Lipschitz control. Maps z in R^K -> scalar energy.
    """
    def __init__(self, num_topics, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * num_topics
        self.fc1 = nn.utils.spectral_norm(nn.Linear(num_topics, hidden_dim))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, z):
        h = F.silu(self.fc1(z))
        h = F.silu(self.fc2(h))
        return self.fc_out(h).squeeze(-1)  # (batch,)


class EBMLTM(nn.Module):
    """
    Energy-Based Latent Topic Model.

    Three components:
    1. Energy-based prior p_alpha(z) ∝ exp(-f_alpha(z)) * N(0,I)
    2. Amortized encoder q_lambda(z|x) used as Langevin initializer
    3. ETM-style decoder p_beta(x|z) = softmax(topic_emb @ word_emb^T) @ theta

    Training phases (handled by EBMLTMTrainer):
    - Warmup (first warmup_epochs): standard VAE (recon + KL)
    - Main: Langevin posterior refinement + energy CD + imitation learning

    Note: training logic lives in EBMLTMTrainer; forward() is for reference only.
    """

    def __init__(
        self,
        vocab_size,
        num_topics=50,
        embed_size=200,
        en_units=512,
        dropout=0.0,
        pretrained_WE=None,
        train_WE=False,
        lv_steps=15,          # posterior Langevin steps
        lv_step_size=0.02,    # posterior Langevin step size
        ls_steps=30,          # prior SGLD steps
        ls_step_size=0.1,     # prior SGLD step size
        warmup_epochs=50,
        kl_weight=1.0,
        au_reg_weight=0.0,    # active-unit regularizer weight (0 = disabled)
        au_tau=0.01,          # variance threshold for AU regularizer
    ):
        super().__init__()

        self.num_topics = num_topics
        self.lv_steps = lv_steps
        self.lv_step_size = lv_step_size
        self.ls_steps = ls_steps
        self.ls_step_size = ls_step_size
        self.warmup_epochs = warmup_epochs
        self.kl_weight = kl_weight
        self.au_reg_weight = au_reg_weight
        self.au_tau = au_tau

        # Persistent epoch counter — included in state_dict
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))

        # ---- Embeddings (ETM-style decoder) ----
        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn(vocab_size, embed_size))
        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, embed_size))

        # ---- Encoder ----
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.ReLU(),
            nn.Linear(en_units, en_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(en_units, num_topics)
        self.fc_logvar = nn.Linear(en_units, num_topics)

        # ---- Energy Network ----
        self.energy_net = EnergyNetwork(num_topics, hidden_dim=2 * num_topics)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode(self, x):
        # Normalize input (ReLU-based encoder requires normalized BoW)
        norm_x = x / (x.sum(1, keepdim=True) + 1e-8)
        h = self.encoder(norm_x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def kl_loss(self, mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

    # ------------------------------------------------------------------
    # Decoder (ETM-style)
    # ------------------------------------------------------------------
    def get_beta(self):
        return F.softmax(
            torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1
        )

    def decode(self, z, beta=None):
        """Decode latent z to reconstruction. Optionally pass pre-computed beta."""
        theta = F.softmax(z, dim=-1)
        if beta is None:
            beta = self.get_beta()
        return torch.matmul(theta, beta), theta

    # ------------------------------------------------------------------
    # Posterior Langevin Refinement
    # ------------------------------------------------------------------
    def langevin_posterior(self, z0, x, num_steps=None, step_size=None):
        """
        Refine z0 toward posterior p(z|x) via Langevin dynamics.
        ∇_z potential = ∇_z f_alpha(z) - ∇_z log p_beta(x|z)

        Optimizations:
        - beta is computed once (not inside the loop)
        - energy_net parameters are frozen during MCMC to avoid graph overhead
        """
        if num_steps is None:
            num_steps = self.lv_steps
        if step_size is None:
            step_size = self.lv_step_size

        if num_steps == 0:
            return z0.detach()

        # Freeze energy_net params during MCMC (only z-gradients needed)
        for p in self.energy_net.parameters():
            p.requires_grad_(False)

        # Pre-compute beta once — doesn't change during Langevin steps
        with torch.no_grad():
            beta = self.get_beta()

        z = z0.detach().clone().requires_grad_(True)

        for _ in range(num_steps):
            e = self.energy_net(z).sum()
            recon = torch.matmul(F.softmax(z, dim=-1), beta)
            log_lik = (x * (recon + 1e-12).log()).sum()
            potential = e - log_lik

            grad = torch.autograd.grad(potential, z)[0]
            noise = torch.randn_like(z) * (step_size ** 0.5)
            z = (z - 0.5 * step_size * grad + noise).detach().requires_grad_(True)

        # Restore energy_net parameter gradients
        for p in self.energy_net.parameters():
            p.requires_grad_(True)

        return z.detach()

    # ------------------------------------------------------------------
    # Prior Short-Run MCMC
    # ------------------------------------------------------------------
    def prior_mcmc(self, num_samples, device, num_steps=None, step_size=None):
        """
        Sample from EBM prior p_alpha(z) via short-run SGLD from N(0,I).
        Freezes energy_net params during sampling to avoid graph overhead.
        """
        if num_steps is None:
            num_steps = self.ls_steps
        if step_size is None:
            step_size = self.ls_step_size

        if num_steps == 0:
            return torch.randn(num_samples, self.num_topics, device=device)

        for p in self.energy_net.parameters():
            p.requires_grad_(False)

        z = torch.randn(num_samples, self.num_topics, device=device).requires_grad_(True)

        for _ in range(num_steps):
            e = self.energy_net(z).sum()
            grad = torch.autograd.grad(e, z)[0]
            noise = torch.randn_like(z) * (step_size ** 0.5)
            z = (z - 0.5 * step_size * grad + noise).detach().requires_grad_(True)

        for p in self.energy_net.parameters():
            p.requires_grad_(True)

        return z.detach()

    # ------------------------------------------------------------------
    # Active-Unit Regularizer
    # ------------------------------------------------------------------
    def au_reg_loss(self, theta):
        """
        Active-Unit regularizer: penalize dimensions with near-zero batch variance.
        Loss = -sum_k min(Var_batch(theta_k), tau)
        Encourages each topic dimension to vary meaningfully across documents.
        """
        if self.au_reg_weight == 0.0:
            return torch.tensor(0.0, device=theta.device)
        var_k = theta.var(dim=0)  # (K,)
        clamped = torch.clamp(var_k, max=self.au_tau)
        return -clamped.sum()

    # ------------------------------------------------------------------
    # VICReg-Style Collapse Prevention
    # ------------------------------------------------------------------
    def vicreg_loss(self, z, lambda_v=0.1, lambda_c=0.1, gamma=1.0):
        """
        VICReg variance + covariance regularization on z (pre-softmax).
        Bardes et al. 2022 applied to topic latent space.

        L_var = (1/K) sum_k max(0, gamma - sqrt(Var(z_k) + eps))
        L_cov = (1/K^2) sum_{i!=j} [Cov(z_i, z_j)]^2

        Args:
            z: (B, K) latent vectors
            lambda_v: variance term weight
            lambda_c: covariance term weight
            gamma: target std (default 1.0)

        Returns:
            vicreg scalar loss
        """
        eps = 1e-4
        B, K = z.shape

        # Center z
        z_c = z - z.mean(dim=0, keepdim=True)

        # Variance term: encourage each dim to have std >= gamma
        std_k = torch.sqrt(z_c.var(dim=0) + eps)  # (K,)
        var_loss = torch.clamp(gamma - std_k, min=0.0).mean()

        # Covariance term: penalize off-diagonal correlations
        cov = (z_c.T @ z_c) / (B - 1)  # (K, K)
        # Zero diagonal before squaring
        diag_mask = torch.eye(K, device=z.device, dtype=torch.bool)
        cov_off = cov.masked_fill(diag_mask, 0.0)
        cov_loss = (cov_off ** 2).sum() / K

        return lambda_v * var_loss + lambda_c * cov_loss

    # ------------------------------------------------------------------
    # get_theta (evaluation — no Langevin by default for speed)
    # ------------------------------------------------------------------
    def get_theta(self, x, refine=False, lv_steps_eval=5):
        mu, logvar = self.encode(x)
        z0 = self.reparameterize(mu, logvar)
        if refine and lv_steps_eval > 0:
            z = self.langevin_posterior(z0, x, num_steps=lv_steps_eval)
        else:
            z = z0
        return F.softmax(z, dim=-1)
