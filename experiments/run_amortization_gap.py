"""
Amortization Gap Measurement.

Measures the empirical amortization gap: how much Langevin refines
the encoder's initial estimate. A large gap means the encoder is
a poor approximate posterior initializer; small gap means it already
approximates the posterior well.

Metrics:
  - L2 distance: E[||z_refined - z_amortized||]
  - ELBO improvement: log p(x|z_refined) - log p(x|z_amortized)
  - Per-dimension variance of (z_refined - z0)
"""

import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmost.data.basic_dataset import BasicDataset
from topmost.models.basic.EBMLTM.EBMLTM import EBMLTM
from topmost.trainers.basic.EBMLTM_trainer import EBMLTMTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

NUM_TOPICS = 50
BATCH_SIZE = 200
SEED = 42
EPOCHS = {'20NG': 200, 'IMDB': 100, 'NYT': 200}


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def measure_gap(ds_name):
    """Train ETM+Langevin on dataset, measure amortization gap on test set."""
    print(f"\n{'='*55}")
    print(f"  Amortization Gap: {ds_name}")
    print(f"{'='*55}")

    path = os.path.join(DATA_DIR, ds_name)
    dataset = BasicDataset(path, batch_size=BATCH_SIZE, read_labels=True, device=DEVICE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    epochs = EPOCHS[ds_name]
    model = EBMLTM(dataset.vocab_size, num_topics=NUM_TOPICS,
                   pretrained_WE=dataset.pretrained_WE,
                   lv_steps=15, ls_steps=0,
                   warmup_epochs=int(epochs * 0.25)).to(DEVICE)
    trainer = EBMLTMTrainer(model, dataset, epochs=epochs, batch_size=BATCH_SIZE)
    trainer.train()

    # Measure gap on test set
    model.eval()
    test_data = dataset.test_data
    data_size = test_data.shape[0]
    all_idx = torch.split(torch.arange(data_size), BATCH_SIZE)

    l2_gaps = []
    loglik_improvements = []
    delta_vars = []

    with torch.no_grad():
        beta = model.get_beta()

    for idx in all_idx:
        batch = test_data[idx]

        with torch.no_grad():
            mu, logvar = model.encode(batch)
            z0 = mu  # Use mean (deterministic) for gap measurement

        # Langevin refinement (allow grad for z)
        z_refined = model.langevin_posterior(z0.detach(), batch)

        with torch.no_grad():
            # L2 gap
            diff = z_refined - z0
            l2 = diff.norm(dim=-1).mean().item()
            l2_gaps.append(l2)

            # Log-likelihood improvement
            theta0 = F.softmax(z0, dim=-1)
            theta_r = F.softmax(z_refined, dim=-1)
            recon0 = torch.matmul(theta0, beta)
            recon_r = torch.matmul(theta_r, beta)
            ll0 = (batch * (recon0 + 1e-12).log()).sum(1).mean().item()
            ll_r = (batch * (recon_r + 1e-12).log()).sum(1).mean().item()
            loglik_improvements.append(ll_r - ll0)

            # Per-dim variance of delta
            delta_vars.append(diff.var(dim=0).cpu().numpy())

    gap_results = {
        'l2_gap_mean': round(float(np.mean(l2_gaps)), 6),
        'l2_gap_std': round(float(np.std(l2_gaps)), 6),
        'loglik_improvement_mean': round(float(np.mean(loglik_improvements)), 6),
        'loglik_improvement_std': round(float(np.std(loglik_improvements)), 6),
        'delta_var_mean': round(float(np.mean(delta_vars)), 6),
    }

    print(f"  L2 gap: {gap_results['l2_gap_mean']:.6f} ± {gap_results['l2_gap_std']:.6f}")
    print(f"  LogLik improvement: {gap_results['loglik_improvement_mean']:.6f} ± {gap_results['loglik_improvement_std']:.6f}")
    print(f"  Delta var: {gap_results['delta_var_mean']:.6f}")

    return gap_results


def main():
    results = load_results()

    for ds_name in ['20NG', 'IMDB', 'NYT']:
        key = f'amort_gap_{ds_name}'
        if key in results:
            print(f"  SKIP {key} (already done: {results[key]})")
            continue
        gap = measure_gap(ds_name)
        results[key] = gap
        save_results(results)

    print("\n" + "="*55)
    print("AMORTIZATION GAP SUMMARY")
    print("="*55)
    print(f"{'Dataset':<10} {'L2 gap':>12} {'ΔlogLik':>14}")
    print("-" * 40)
    for ds_name in ['20NG', 'IMDB', 'NYT']:
        key = f'amort_gap_{ds_name}'
        r = results.get(key, {})
        if r:
            print(f"  {ds_name:<8} {r['l2_gap_mean']:>12.6f} {r['loglik_improvement_mean']:>14.6f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
