"""
EBM-LTM Trainer

Implements Algorithm 1 from the EBM-LTM proposal with:
- Warmup phase: standard VAE to prevent early posterior collapse
- Main phase: energy CD + Langevin refinement + imitation learning
- Separate optimizers for energy (1e-4) and rest (2e-3)
"""

import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from topmost.utils import _utils
from topmost.utils.logger import Logger

logger = Logger("WARNING")


class EBMLTMTrainer:
    def __init__(
        self,
        model,
        dataset,
        num_top_words=15,
        epochs=200,
        learning_rate=2e-3,
        energy_learning_rate=1e-4,
        batch_size=200,
        log_interval=10,
        verbose=False,
        lambda_v=0.0,    # VICReg variance term weight (0 = disabled)
        lambda_c=0.0,    # VICReg covariance term weight (0 = disabled)
        vicreg_gamma=1.0,  # VICReg target std
    ):
        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.energy_learning_rate = energy_learning_rate
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.verbose = verbose
        self.lambda_v = lambda_v
        self.lambda_c = lambda_c
        self.vicreg_gamma = vicreg_gamma

        if verbose:
            logger.set_level("DEBUG")

    def make_optimizers(self):
        energy_params = list(self.model.energy_net.parameters())
        energy_param_ids = set(id(p) for p in energy_params)
        other_params = [p for p in self.model.parameters()
                        if id(p) not in energy_param_ids]

        optimizer_main = torch.optim.Adam(other_params, lr=self.learning_rate)
        optimizer_energy = torch.optim.Adam(energy_params, lr=self.energy_learning_rate)
        return optimizer_main, optimizer_energy

    def train(self):
        optimizer_main, optimizer_energy = self.make_optimizers()
        data_size = len(self.dataset.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            self.model.current_epoch = torch.tensor(epoch, dtype=torch.long)
            loss_rst_dict = defaultdict(float)
            # Use <= to match model.forward() reference; warmup_epochs is inclusive
            in_warmup = epoch <= self.model.warmup_epochs

            for batch_data in self.dataset.train_dataloader:
                B = batch_data.shape[0]
                device = batch_data.device

                if in_warmup:
                    # --- Warmup: standard VAE + AU reg + VICReg ---
                    mu, logvar = self.model.encode(batch_data)
                    z0 = self.model.reparameterize(mu, logvar)
                    recon, theta = self.model.decode(z0)
                    recon_loss = -(batch_data * (recon + 1e-12).log()).sum(1).mean()
                    kl = self.model.kl_loss(mu, logvar)
                    au_loss = self.model.au_reg_loss(theta)
                    loss = recon_loss + self.model.kl_weight * kl + self.model.au_reg_weight * au_loss
                    if self.lambda_v > 0 or self.lambda_c > 0:
                        vicreg = self.model.vicreg_loss(z0, self.lambda_v, self.lambda_c, self.vicreg_gamma)
                        loss = loss + vicreg
                        loss_rst_dict['vicreg_loss'] += vicreg.item() * B

                    optimizer_main.zero_grad()
                    loss.backward()
                    optimizer_main.step()

                    loss_rst_dict['recon_loss'] += recon_loss.item() * B
                    loss_rst_dict['kl_loss'] += kl.item() * B
                    if self.model.au_reg_weight > 0:
                        loss_rst_dict['au_loss'] += au_loss.item() * B

                else:
                    # --- Main phase: EBM + Langevin ---

                    # Step 1: Amortized init
                    mu, logvar = self.model.encode(batch_data)
                    z0 = self.model.reparameterize(mu, logvar)

                    # Step 2: Posterior Langevin refinement (no grad needed)
                    z_refined = self.model.langevin_posterior(z0.detach(), batch_data)

                    # Step 3: Decoder + KL + AU + VICReg update (main optimizer)
                    recon, theta = self.model.decode(z_refined.detach())
                    recon_loss = -(batch_data * (recon + 1e-12).log()).sum(1).mean()
                    kl = self.model.kl_loss(mu, logvar)
                    init_loss = F.mse_loss(z0, z_refined.detach())
                    au_loss = self.model.au_reg_loss(theta)
                    main_loss = (recon_loss + 0.1 * kl + init_loss
                                 + self.model.au_reg_weight * au_loss)
                    if self.lambda_v > 0 or self.lambda_c > 0:
                        # Apply VICReg to refined z for collapse prevention
                        vicreg = self.model.vicreg_loss(z_refined.detach(), self.lambda_v, self.lambda_c, self.vicreg_gamma)
                        main_loss = main_loss + vicreg
                        loss_rst_dict['vicreg_loss'] += vicreg.item() * B

                    optimizer_main.zero_grad()
                    main_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters()
                         if id(p) not in set(id(q) for q in self.model.energy_net.parameters())],
                        5.0
                    )
                    optimizer_main.step()

                    # Step 4: Energy contrastive divergence (energy optimizer)
                    z_prior = self.model.prior_mcmc(B, device)
                    energy_pos = self.model.energy_net(z_refined.detach()).mean()
                    energy_neg = self.model.energy_net(z_prior).mean()
                    energy_loss = energy_pos - energy_neg

                    optimizer_energy.zero_grad()
                    energy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.energy_net.parameters(), 1.0)
                    optimizer_energy.step()

                    loss_rst_dict['recon_loss'] += recon_loss.item() * B
                    loss_rst_dict['kl_loss'] += kl.item() * B
                    loss_rst_dict['energy_loss'] += energy_loss.item() * B
                    loss_rst_dict['init_loss'] += init_loss.item() * B
                    if self.model.au_reg_weight > 0:
                        loss_rst_dict['au_loss'] += au_loss.item() * B

            if epoch % self.log_interval == 0:
                phase = 'warmup' if in_warmup else 'main'
                output_log = f'Epoch: {epoch:03d} [{phase}]'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size:.4f}'
                logger.info(output_log)

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_data)
        return top_words, train_theta

    def test(self, bow):
        data_size = bow.shape[0]
        theta = []
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = bow[idx]
                batch_theta = self.model.get_theta(batch_input, refine=False)
                theta.extend(batch_theta.cpu().tolist())

        return np.asarray(theta)

    def get_beta(self):
        return self.model.get_beta().detach().cpu().numpy()

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        beta = self.get_beta()
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words, self.verbose)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_data)
        test_theta = self.test(self.dataset.test_data)
        return train_theta, test_theta

    def held_out_perplexity(self, test_data=None):
        """
        Compute held-out perplexity two ways:
          - PPL-amortized: using encoder mean z0 (no Langevin)
          - PPL-refined:   using Langevin-refined z (lv_steps posterior steps)

        PPL = exp( - (1/N) sum_n (1/|d_n|) sum_v x_{n,v} log p(x_{n,v}|z_n) )

        Returns dict with keys 'ppl_amortized' and 'ppl_refined'.
        """
        if test_data is None:
            test_data = self.dataset.test_data

        self.model.eval()
        all_idx = torch.split(torch.arange(test_data.shape[0]), self.batch_size)

        nll_amortized = 0.0
        nll_refined = 0.0
        total_tokens = 0

        with torch.no_grad():
            beta = self.model.get_beta()

        for idx in all_idx:
            batch = test_data[idx]
            n_tokens = batch.sum().item()
            total_tokens += n_tokens

            with torch.no_grad():
                mu, logvar = self.model.encode(batch)
                z0 = mu  # deterministic amortized

            # Amortized reconstruction
            with torch.no_grad():
                recon0 = torch.matmul(F.softmax(z0, dim=-1), beta)
                ll0 = (batch * (recon0 + 1e-12).log()).sum().item()
                nll_amortized -= ll0

            # Langevin-refined reconstruction
            z_r = self.model.langevin_posterior(z0.detach(), batch)
            with torch.no_grad():
                recon_r = torch.matmul(F.softmax(z_r, dim=-1), beta)
                ll_r = (batch * (recon_r + 1e-12).log()).sum().item()
                nll_refined -= ll_r

        ppl_amortized = float(np.exp(nll_amortized / total_tokens))
        ppl_refined = float(np.exp(nll_refined / total_tokens))
        return {'ppl_amortized': round(ppl_amortized, 2), 'ppl_refined': round(ppl_refined, 2)}
