"""
Step Size Ablation for EBM-LTM Paper.

ETM+Langevin with varying Langevin step size delta_v on 20NG.
Reports: NPMI, TD, AU, Acc, exploration distance ||z_Lv - z0||_2
Saves to experiments/results.json with keys: stepsize_{delta}_20NG
"""

import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmost.data.basic_dataset import BasicDataset
from topmost.models.basic.EBMLTM.EBMLTM import EBMLTM
from topmost.trainers.basic.EBMLTM_trainer import EBMLTMTrainer
from topmost.eva.topic_coherence import _coherence
from topmost.eva.topic_diversity import _diversity
from topmost.eva.classification import _cls

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

NUM_TOPICS = 50
BATCH_SIZE = 200
NUM_TOP_WORDS = 15
COHERENCE_TOPN = 10
SEED = 42
EPOCHS = 200
DS_NAME = '20NG'

DELTA_V_GRID = [0.01, 0.02, 0.05, 0.1, 0.2]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def measure_exploration_distance(model, dataset):
    """
    Compute mean L2 distance ||z_Lv - z0||_2 on the test set.
    Uses trained model's Langevin posterior.
    """
    model.eval()
    test_data = dataset.test_data
    all_idx = torch.split(torch.arange(test_data.shape[0]), BATCH_SIZE)
    l2_gaps = []

    for idx in all_idx:
        with torch.no_grad():
            batch = test_data[idx]
            mu, logvar = model.encode(batch)
            z0 = mu  # deterministic
        # langevin_posterior needs grad tracking enabled
        z_r = model.langevin_posterior(z0.detach(), batch)
        with torch.no_grad():
            diff = z_r - z0.detach()
            l2 = diff.norm(dim=-1).mean().item()
            l2_gaps.append(l2)

    return round(float(np.mean(l2_gaps)), 6)


def evaluate(trainer, dataset, model):
    top_words = trainer.get_top_words(NUM_TOP_WORDS)
    train_theta, test_theta = trainer.export_theta()

    try:
        npmi = _coherence(dataset.train_texts, dataset.vocab, top_words,
                          coherence_type='c_npmi', topn=COHERENCE_TOPN)
    except:
        npmi = float('nan')

    td = _diversity(top_words)

    try:
        cls_r = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
        acc, f1 = cls_r['acc'], cls_r['macro-F1']
    except:
        acc, f1 = float('nan'), float('nan')

    au = int(np.sum(np.var(train_theta, axis=0) > 0.01))
    explore_dist = measure_exploration_distance(model, dataset)

    return {
        'NPMI': round(float(npmi), 4),
        'TD': round(float(td), 4),
        'NPMI_x_TD': round(float(npmi * td), 4) if not np.isnan(npmi) else float('nan'),
        'Acc': round(float(acc), 4),
        'F1': round(float(f1), 4),
        'AU': au,
        'explore_dist': explore_dist,
    }


def run_stepsize(delta_v, dataset, results):
    key = f'stepsize_{delta_v}_{DS_NAME}'
    if key in results and 'NPMI' in results[key]:
        print(f"  SKIP {key}")
        return results[key]

    print(f"\n{'='*55}")
    print(f"  StepSize δ_v={delta_v} | {DS_NAME} | seed={SEED}")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = EBMLTM(dataset.vocab_size, num_topics=NUM_TOPICS,
                   pretrained_WE=dataset.pretrained_WE,
                   lv_steps=15, lv_step_size=delta_v,
                   ls_steps=0,  # ETM+Langevin (no EBM prior)
                   warmup_epochs=int(EPOCHS * 0.25),
                   kl_weight=1.0).to(DEVICE)
    trainer = EBMLTMTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                            epochs=EPOCHS, batch_size=BATCH_SIZE)

    t0 = time.time()
    try:
        trainer.train()
        elapsed = round(time.time() - t0, 1)
        r = evaluate(trainer, dataset, model)
        r['time_sec'] = elapsed
        r['delta_v'] = delta_v
        print(f"  DONE: NPMI={r['NPMI']:.4f} TD={r['TD']:.4f} AU={r['AU']} "
              f"explore={r['explore_dist']:.4f} t={elapsed}s")
    except Exception as e:
        import traceback; traceback.print_exc()
        r = {'error': str(e), 'delta_v': delta_v}
        print(f"  ERROR: {e}")

    results[key] = r
    save_results(results)
    return r


def print_summary(results):
    print("\n" + "="*70)
    print(f"STEP SIZE ABLATION SUMMARY ({DS_NAME}, ETM+Langevin, seed={SEED})")
    print("="*70)
    hdr = f"{'δ_v':>6} {'NPMI':>8} {'TD':>7} {'AU':>4} {'Acc':>7} {'Explore||z||':>14}"
    print(hdr)
    print("-" * len(hdr))
    for delta_v in DELTA_V_GRID:
        k = f'stepsize_{delta_v}_{DS_NAME}'
        r = results.get(k, {})
        if 'error' in r or not r:
            print(f"  {delta_v:>4}  (error or missing)")
            continue
        print(f"  {delta_v:>4} "
              f"{r.get('NPMI', float('nan')):>8.4f} "
              f"{r.get('TD', float('nan')):>7.4f} "
              f"{r.get('AU', 0):>4d} "
              f"{r.get('Acc', float('nan')):>7.4f} "
              f"{r.get('explore_dist', float('nan')):>14.6f}")


def main():
    results = load_results()

    path = os.path.join(DATA_DIR, DS_NAME)
    dataset = BasicDataset(path, batch_size=BATCH_SIZE, read_labels=True, device=DEVICE)

    for delta_v in DELTA_V_GRID:
        run_stepsize(delta_v, dataset, results)

    print_summary(results)
    print("\nDone. Results saved to results.json")


if __name__ == '__main__':
    main()
