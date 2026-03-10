import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import StepLR
from topmost.utils import _utils
from topmost.utils.logger import Logger

logger = Logger("WARNING")


class BasicTrainer:
    def __init__(self,
                 model,
                 dataset,
                 num_top_words=15,
                 epochs=200,
                 learning_rate=0.002,
                 batch_size=200,
                 lr_scheduler=None,
                 lr_step_size=125,
                 log_interval=5,
                 verbose=False
                ):

        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.verbose = verbose
        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            logger.info("use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in self.dataset.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                logger.info(output_log)

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_data)

        return top_words, train_theta

    def test(self, bow):
        data_size = bow.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = bow[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def get_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

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

    def test_time_langevin_refine(self, bow, n_steps=15, step_size=0.02):
        """Apply Langevin refinement at test time. Model must be in eval mode.
        Returns theta as numpy array."""
        self.model.eval()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size), self.batch_size)
        thetas = []

        for idx in all_idx:
            batch = bow[idx]
            with torch.no_grad():
                # Get encoder mu (deterministic in eval mode)
                mu = self.model.get_theta(batch)
                # For ETM, get_theta returns softmax(mu) in eval;
                # we need the pre-softmax z, so use encode directly
                if hasattr(self.model, 'encode'):
                    enc_out = self.model.encode(batch / batch.sum(1, keepdim=True)
                                                if hasattr(self.model, 'word_embeddings')
                                                and not hasattr(self.model, 'mean_bn')
                                                else batch)
                    if isinstance(enc_out, tuple):
                        z_init = enc_out[0]  # mu
                    else:
                        z_init = enc_out
                else:
                    # Fallback: invert softmax approximately
                    z_init = mu.log()

            z = z_init.clone().detach().requires_grad_(True)
            for _ in range(n_steps):
                with torch.no_grad():
                    beta = self.model.get_beta()  # [K, V]
                theta = torch.softmax(z, dim=-1)  # [B, K]
                recon = torch.matmul(theta, beta)  # [B, V]
                log_lik = (batch * (recon + 1e-10).log()).sum(-1)
                grad = torch.autograd.grad(log_lik.sum(), z)[0]
                grad = grad - z  # Gaussian prior gradient
                noise = torch.randn_like(z) * (step_size ** 0.5)
                z = (z.detach() + 0.5 * step_size * grad + noise).requires_grad_(True)

            thetas.append(torch.softmax(z.detach(), dim=-1).cpu())

        return torch.cat(thetas, dim=0).numpy()

    def _get_encoder_z(self, batch):
        """Get pre-softmax z from encoder for a batch (handles ETM and ECRTM)."""
        if hasattr(self.model, 'mean_bn'):
            # ECRTM style: encode returns (theta, loss_KL), need to get mu directly
            e1 = torch.nn.functional.softplus(self.model.fc11(batch))
            e1 = torch.nn.functional.softplus(self.model.fc12(e1))
            e1 = self.model.fc1_dropout(e1)
            mu = self.model.mean_bn(self.model.fc21(e1))
            return mu
        else:
            # ETM style: normalize input, encode returns (mu, logvar)
            norm_x = batch / batch.sum(1, keepdim=True)
            mu, logvar = self.model.encode(norm_x)
            return mu

    def held_out_perplexity_refined(self, test_data=None, n_steps=15, step_size=0.02):
        """PPL using test-time Langevin refined z."""
        if test_data is None:
            test_data = self.dataset.test_data
        self.model.eval()
        all_idx = torch.split(torch.arange(test_data.shape[0]), self.batch_size)
        nll = 0.0
        total_tokens = 0

        for idx in all_idx:
            batch = test_data[idx]
            n_tokens = batch.sum().item()
            total_tokens += n_tokens

            with torch.no_grad():
                mu = self._get_encoder_z(batch)

            z = mu.clone().detach().requires_grad_(True)
            for _ in range(n_steps):
                with torch.no_grad():
                    beta = self.model.get_beta()
                theta = torch.softmax(z, dim=-1)
                recon = torch.matmul(theta, beta)
                log_lik = (batch * (recon + 1e-10).log()).sum(-1)
                grad = torch.autograd.grad(log_lik.sum(), z)[0]
                grad = grad - z
                noise = torch.randn_like(z) * (step_size ** 0.5)
                z = (z.detach() + 0.5 * step_size * grad + noise).requires_grad_(True)

            with torch.no_grad():
                beta = self.model.get_beta()
                theta = torch.softmax(z.detach(), dim=-1)
                recon = torch.matmul(theta, beta)
                ll = (batch * (recon + 1e-10).log()).sum().item()
                nll -= ll

        ppl = float(np.exp(nll / total_tokens))
        return round(ppl, 2)

    def held_out_perplexity(self, test_data=None):
        """
        Compute held-out perplexity using amortized encoder (no MCMC).
        PPL = exp( -(1/N_tokens) * sum log p(x|z_amortized) )
        Returns dict with key 'ppl_amortized'.
        """
        if test_data is None:
            test_data = self.dataset.test_data

        self.model.eval()
        all_idx = torch.split(torch.arange(test_data.shape[0]), self.batch_size)

        nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for idx in all_idx:
                batch = test_data[idx]
                n_tokens = batch.sum().item()
                total_tokens += n_tokens

                theta = self.model.get_theta(batch)
                if hasattr(self.model, 'get_recon'):
                    recon = self.model.get_recon(theta)
                else:
                    beta = self.model.get_beta()
                    recon = torch.matmul(theta, beta)
                ll = (batch * (recon + 1e-12).log()).sum().item()
                nll -= ll

        ppl = float(np.exp(nll / total_tokens))
        return {'ppl_amortized': round(ppl, 2), 'ppl_refined': None}
