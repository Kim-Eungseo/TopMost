"""
VICReg Ablation for EBM-LTM Paper.

Grid search: lambda_v x lambda_c on 20NG, then best config on NYT + multi-seed.
Saves to experiments/results.json with keys:
  vicreg_{lv}_{lc}_20NG   (single seed=42)
  vicreg_best_{dataset}_seed{seed}
  vicreg_agg_{model}_{dataset}
"""

import sys, os, json, time
import numpy as np
import torch

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
SEEDS = [42, 123, 456]
EPOCHS = {'20NG': 200, 'NYT': 200}

# Grid search space
LV_GRID = [0.01, 0.1, 0.5, 1.0]
LC_GRID = [0.01, 0.1, 0.5]
GAMMA = 1.0


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_dataset(name):
    path = os.path.join(DATA_DIR, name)
    return BasicDataset(path, batch_size=BATCH_SIZE, read_labels=True, device=DEVICE)


def make_model_trainer(dataset, epochs, seed, lambda_v, lambda_c):
    vocab_size = dataset.vocab_size
    we = dataset.pretrained_WE
    model = EBMLTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we,
                   lv_steps=15, ls_steps=30,
                   warmup_epochs=int(epochs * 0.25),
                   kl_weight=1.0).to(DEVICE)
    trainer = EBMLTMTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                            epochs=epochs, batch_size=BATCH_SIZE,
                            lambda_v=lambda_v, lambda_c=lambda_c, vicreg_gamma=GAMMA)
    return model, trainer


def evaluate(trainer, dataset):
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

    return {
        'NPMI': round(float(npmi), 4),
        'TD': round(float(td), 4),
        'NPMI_x_TD': round(float(npmi * td), 4) if not np.isnan(npmi) else float('nan'),
        'Acc': round(float(acc), 4),
        'F1': round(float(f1), 4),
        'AU': au,
    }


def run_grid_search(results):
    """Run λ_v × λ_c grid search on 20NG with seed=42."""
    ds_name = '20NG'
    dataset = load_dataset(ds_name)
    epochs = EPOCHS[ds_name]

    best_score = -float('inf')
    best_config = None

    for lv in LV_GRID:
        for lc in LC_GRID:
            key = f'vicreg_{lv}_{lc}_{ds_name}'
            if key in results and 'NPMI' in results[key]:
                print(f"  SKIP {key}")
                r = results[key]
            else:
                print(f"\n{'='*55}")
                print(f"  VICReg | λ_v={lv} λ_c={lc} | {ds_name} | seed={SEED}")
                print(f"{'='*55}")

                torch.manual_seed(SEED)
                np.random.seed(SEED)

                model, trainer = make_model_trainer(dataset, epochs, SEED, lv, lc)
                t0 = time.time()
                try:
                    trainer.train()
                    elapsed = round(time.time() - t0, 1)
                    r = evaluate(trainer, dataset)
                    r['time_sec'] = elapsed
                    r['lambda_v'] = lv
                    r['lambda_c'] = lc
                    print(f"  DONE: NPMI={r['NPMI']:.4f} TD={r['TD']:.4f} AU={r['AU']} Acc={r['Acc']:.4f} t={elapsed}s")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    r = {'error': str(e), 'lambda_v': lv, 'lambda_c': lc}
                    print(f"  ERROR: {e}")

                results[key] = r
                save_results(results)

            # Track best by NPMI×TD (primary) + AU as tiebreaker
            if 'NPMI_x_TD' in r and not isinstance(r.get('NPMI_x_TD'), str):
                score = r['NPMI_x_TD'] + 0.001 * r.get('AU', 0)
                if score > best_score:
                    best_score = score
                    best_config = (lv, lc)

    print(f"\n  Best config: λ_v={best_config[0]}, λ_c={best_config[1]} (score={best_score:.6f})")
    results['vicreg_best_config'] = {'lambda_v': best_config[0], 'lambda_c': best_config[1],
                                      'score': best_score}
    save_results(results)
    return best_config


def run_best_config(results, best_config):
    """Run best VICReg config on 20NG + NYT with 3 seeds."""
    lv, lc = best_config
    print(f"\n\nRunning best config (λ_v={lv}, λ_c={lc}) multi-seed on 20NG + NYT")

    for ds_name in ['20NG', 'NYT']:
        dataset = load_dataset(ds_name)
        epochs = EPOCHS[ds_name]
        seed_results = []

        for seed in SEEDS:
            key = f'vicreg_best_{ds_name}_seed{seed}'
            if key in results and 'NPMI' in results[key]:
                print(f"  SKIP {key}")
                seed_results.append(results[key])
                continue

            print(f"\n{'='*55}")
            print(f"  VICReg Best | {ds_name} | seed={seed}")
            print(f"{'='*55}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            model, trainer = make_model_trainer(dataset, epochs, seed, lv, lc)
            t0 = time.time()
            try:
                trainer.train()
                elapsed = round(time.time() - t0, 1)
                r = evaluate(trainer, dataset)
                r['time_sec'] = elapsed
                r['lambda_v'] = lv
                r['lambda_c'] = lc
                print(f"  DONE: NPMI={r['NPMI']:.4f} TD={r['TD']:.4f} AU={r['AU']} Acc={r['Acc']:.4f} t={elapsed}s")
                seed_results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                r = {'error': str(e)}
                print(f"  ERROR: {e}")

            results[key] = r
            save_results(results)

        # Aggregate
        valid = [r for r in seed_results if 'NPMI' in r and 'error' not in r]
        if valid:
            agg = {}
            for metric in ['NPMI', 'TD', 'NPMI_x_TD', 'Acc', 'F1', 'AU']:
                vals = [r[metric] for r in valid if metric in r]
                if vals:
                    agg[f'{metric}_mean'] = round(float(np.mean(vals)), 4)
                    agg[f'{metric}_std'] = round(float(np.std(vals)), 4)
            agg['lambda_v'] = lv
            agg['lambda_c'] = lc
            agg_key = f'vicreg_agg_{ds_name}'
            results[agg_key] = agg
            save_results(results)
            print(f"\n  {ds_name} aggregate: NPMI={agg['NPMI_mean']:.4f}±{agg['NPMI_std']:.4f} AU={agg['AU_mean']:.1f}")


def print_summary(results):
    print("\n" + "="*70)
    print("VICREG ABLATION SUMMARY (20NG, seed=42)")
    print("="*70)
    hdr = f"{'λ_v':>6} {'λ_c':>6} {'NPMI':>8} {'TD':>7} {'NPMI×TD':>9} {'AU':>4} {'Acc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for lv in LV_GRID:
        for lc in LC_GRID:
            k = f'vicreg_{lv}_{lc}_20NG'
            r = results.get(k, {})
            if 'error' in r or not r:
                continue
            print(f"  {lv:>4} {lc:>6} "
                  f"{r.get('NPMI', float('nan')):>8.4f} "
                  f"{r.get('TD', float('nan')):>7.4f} "
                  f"{r.get('NPMI_x_TD', float('nan')):>9.4f} "
                  f"{r.get('AU', 0):>4d} "
                  f"{r.get('Acc', float('nan')):>7.4f}")

    print("\nBest config multi-seed results:")
    for ds_name in ['20NG', 'NYT']:
        agg = results.get(f'vicreg_agg_{ds_name}', {})
        if agg:
            print(f"  {ds_name}: NPMI={agg.get('NPMI_mean', 'n/a'):.4f}±{agg.get('NPMI_std', 0):.4f} "
                  f"AU={agg.get('AU_mean', 'n/a'):.1f}")


def main():
    results = load_results()

    # Phase 1: Grid search on 20NG
    best_config = run_grid_search(results)

    # Phase 2: Best config multi-seed on 20NG + NYT
    run_best_config(results, best_config)

    print_summary(results)
    print("\nDone. Results saved to results.json")


if __name__ == '__main__':
    main()
