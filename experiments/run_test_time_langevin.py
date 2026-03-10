"""
Test-Time Langevin (TTL) Experiment Runner

Solution A: Train a standard ETM/ECRTM (unchanged ELBO), apply Langevin
refinement ONLY at test/inference time for topic extraction.

Key properties:
- PPL-amort = ETM's PPL (same trained model)
- PPL-refined <= PPL-amort (guaranteed)
- Training cost = ETM
- Story: "Langevin refinement is a zero-cost inference-time module"

Langevin update (Gaussian prior):
  z_{t+1} = z_t + (dv/2) * [grad_z log p_beta(x|z_t) - z_t] + sqrt(dv) * eps_t

Saves to experiments/results.json with keys like ttl_ETM_lv15_dv0.02_20NG
"""

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmost.data.basic_dataset import BasicDataset
from topmost.models.basic.ETM import ETM
from topmost.models.basic.ECRTM.ECRTM import ECRTM
from topmost.trainers.basic.basic_trainer import BasicTrainer
from topmost.eva.topic_coherence import _coherence
from topmost.eva.topic_diversity import _diversity
from topmost.eva.classification import _cls

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

NUM_TOPICS = 50
BATCH_SIZE = 200
SEED = 42
NUM_TOP_WORDS = 15
COHERENCE_TOPN = 10

EPOCHS = {'20NG': 200, 'IMDB': 200, 'NYT': 200}
DATASETS = ['20NG', 'NYT', 'IMDB']

# Grid sweep
N_STEPS_GRID = [5, 10, 15, 20]
STEP_SIZE_GRID = [0.01, 0.02, 0.05]


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
    print(f"\nLoading {name}...")
    ds = BasicDataset(path, batch_size=BATCH_SIZE, read_labels=True, device=DEVICE)
    return ds


def evaluate_with_ttl(trainer, dataset, n_steps, step_size):
    """Evaluate using test-time Langevin refined theta."""
    top_words = trainer.get_top_words(NUM_TOP_WORDS)

    # Get TTL-refined theta for train and test
    print(f"    Computing TTL theta (train)...")
    train_theta = trainer.test_time_langevin_refine(
        dataset.train_data, n_steps=n_steps, step_size=step_size)
    print(f"    Computing TTL theta (test)...")
    test_theta = trainer.test_time_langevin_refine(
        dataset.test_data, n_steps=n_steps, step_size=step_size)

    # NPMI coherence
    try:
        npmi = _coherence(dataset.train_texts, dataset.vocab, top_words,
                          coherence_type='c_npmi', topn=COHERENCE_TOPN)
    except Exception as e:
        print(f"  NPMI failed ({e})")
        npmi = float('nan')

    td = _diversity(top_words)

    try:
        cls_r = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
        acc, f1 = cls_r['acc'], cls_r['macro-F1']
    except Exception as e:
        print(f"  Classification failed: {e}")
        acc, f1 = float('nan'), float('nan')

    au = int(np.sum(np.var(train_theta, axis=0) > 0.01))

    return {
        'NPMI': round(npmi, 4),
        'TD': round(td, 4),
        'NPMI_x_TD': round(npmi * td, 4) if not np.isnan(npmi) else float('nan'),
        'Acc': round(acc, 4),
        'F1': round(f1, 4),
        'AU': au,
    }


def evaluate_baseline(trainer, dataset):
    """Evaluate using standard amortized theta (no Langevin)."""
    top_words = trainer.get_top_words(NUM_TOP_WORDS)
    train_theta = trainer.test(dataset.train_data)
    test_theta = trainer.test(dataset.test_data)

    try:
        npmi = _coherence(dataset.train_texts, dataset.vocab, top_words,
                          coherence_type='c_npmi', topn=COHERENCE_TOPN)
    except Exception as e:
        print(f"  NPMI failed ({e})")
        npmi = float('nan')

    td = _diversity(top_words)

    try:
        cls_r = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
        acc, f1 = cls_r['acc'], cls_r['macro-F1']
    except Exception as e:
        print(f"  Classification failed: {e}")
        acc, f1 = float('nan'), float('nan')

    au = int(np.sum(np.var(train_theta, axis=0) > 0.01))

    return {
        'NPMI': round(npmi, 4),
        'TD': round(td, 4),
        'NPMI_x_TD': round(npmi * td, 4) if not np.isnan(npmi) else float('nan'),
        'Acc': round(acc, 4),
        'F1': round(f1, 4),
        'AU': au,
    }


def train_and_sweep(model_name, ds_name, dataset, results):
    """Train a model once, then sweep TTL configurations."""
    epochs = EPOCHS[ds_name]
    vocab_size = dataset.vocab_size
    we = dataset.pretrained_WE

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create model
    if model_name == 'ETM':
        model = ETM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
    elif model_name == 'ECRTM':
        model = ECRTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    trainer = BasicTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                           epochs=epochs, batch_size=BATCH_SIZE)

    # Train
    print(f"\n{'='*60}")
    print(f"  Training {model_name} on {ds_name} ({epochs} epochs)")
    print(f"{'='*60}")
    t0 = time.time()
    trainer.train()
    train_time = round(time.time() - t0, 1)
    print(f"  Training done in {train_time}s")

    # Baseline evaluation (amortized)
    baseline_key = f'ttl_{model_name}_baseline_{ds_name}'
    if baseline_key not in results or 'NPMI' not in results.get(baseline_key, {}):
        print(f"\n  Evaluating baseline (amortized)...")
        baseline_metrics = evaluate_baseline(trainer, dataset)
        baseline_metrics['train_time_sec'] = train_time

        # Compute amortized PPL
        ppl_dict = trainer.held_out_perplexity()
        baseline_metrics['PPL_amort'] = ppl_dict['ppl_amortized']
        baseline_metrics['PPL_refined'] = None

        results[baseline_key] = baseline_metrics
        save_results(results)
        print(f"  Baseline: NPMI={baseline_metrics['NPMI']:.4f} TD={baseline_metrics['TD']:.4f} "
              f"Acc={baseline_metrics['Acc']:.4f} PPL={baseline_metrics['PPL_amort']:.2f}")
    else:
        print(f"  SKIP baseline {baseline_key} (already done)")

    # Sweep TTL configurations
    for n_steps in N_STEPS_GRID:
        for step_size in STEP_SIZE_GRID:
            key = f'ttl_{model_name}_lv{n_steps}_dv{step_size}_{ds_name}'
            if key in results and 'NPMI' in results.get(key, {}):
                print(f"  SKIP {key} (already done)")
                continue

            print(f"\n  TTL: n_steps={n_steps}, step_size={step_size}")
            t1 = time.time()
            try:
                ttl_metrics = evaluate_with_ttl(trainer, dataset, n_steps, step_size)
                ttl_time = round(time.time() - t1, 1)
                ttl_metrics['train_time_sec'] = train_time
                ttl_metrics['ttl_time_sec'] = ttl_time
                ttl_metrics['n_steps'] = n_steps
                ttl_metrics['step_size'] = step_size
                ttl_metrics['dataset'] = ds_name
                ttl_metrics['base_model'] = model_name

                # Compute PPL-amort (same model, same value for all configs)
                ppl_amort = results.get(baseline_key, {}).get('PPL_amort', None)
                if ppl_amort is None:
                    ppl_dict = trainer.held_out_perplexity()
                    ppl_amort = ppl_dict['ppl_amortized']
                ttl_metrics['PPL_amort'] = ppl_amort

                # Compute PPL-refined with this TTL config
                print(f"    Computing PPL-refined...")
                ppl_refined = trainer.held_out_perplexity_refined(
                    n_steps=n_steps, step_size=step_size)
                ttl_metrics['PPL_refined'] = ppl_refined

                print(f"  DONE: NPMI={ttl_metrics['NPMI']:.4f} TD={ttl_metrics['TD']:.4f} "
                      f"Acc={ttl_metrics['Acc']:.4f} PPL-a={ppl_amort:.2f} PPL-r={ppl_refined:.2f} "
                      f"t={ttl_time}s")

            except Exception as e:
                import traceback; traceback.print_exc()
                ttl_metrics = {'error': str(e), 'n_steps': n_steps,
                               'step_size': step_size, 'dataset': ds_name}
                print(f"  ERROR: {e}")

            results[key] = ttl_metrics
            save_results(results)

    return results


def find_best_config(results, model_name, ds_name):
    """Find the best TTL config by NPMI for a given model and dataset."""
    best_key = None
    best_npmi = -float('inf')
    for n_steps in N_STEPS_GRID:
        for step_size in STEP_SIZE_GRID:
            key = f'ttl_{model_name}_lv{n_steps}_dv{step_size}_{ds_name}'
            r = results.get(key, {})
            npmi = r.get('NPMI', -float('inf'))
            if not np.isnan(npmi) and npmi > best_npmi:
                best_npmi = npmi
                best_key = key
    return best_key


def print_summary(results):
    print("\n" + "="*70)
    print("TEST-TIME LANGEVIN RESULTS SUMMARY")
    print("="*70)

    for model_name in ['ETM', 'ECRTM']:
        print(f"\n--- {model_name} ---")
        for ds_name in DATASETS:
            print(f"\n  Dataset: {ds_name}")
            # Baseline
            bk = f'ttl_{model_name}_baseline_{ds_name}'
            br = results.get(bk, {})
            if br and 'NPMI' in br:
                print(f"    Baseline: NPMI={br['NPMI']:.4f} TD={br['TD']:.4f} "
                      f"Acc={br['Acc']:.4f} PPL={br.get('PPL_amort', 'N/A')}")

            # Grid
            print(f"    {'n_steps':>7} {'step_size':>10} {'NPMI':>8} {'TD':>7} "
                  f"{'Acc':>7} {'PPL-a':>8} {'PPL-r':>8}")
            print(f"    {'-'*58}")
            for n_steps in N_STEPS_GRID:
                for step_size in STEP_SIZE_GRID:
                    key = f'ttl_{model_name}_lv{n_steps}_dv{step_size}_{ds_name}'
                    r = results.get(key, {})
                    if r and 'NPMI' in r:
                        print(f"    {n_steps:>7} {step_size:>10.2f} "
                              f"{r['NPMI']:>8.4f} {r['TD']:>7.4f} "
                              f"{r['Acc']:>7.4f} "
                              f"{r.get('PPL_amort', 0):>8.2f} "
                              f"{r.get('PPL_refined', 0):>8.2f}")

            # Best config
            best_key = find_best_config(results, model_name, ds_name)
            if best_key:
                br2 = results[best_key]
                print(f"    BEST: {best_key} -> NPMI={br2['NPMI']:.4f}")


def main():
    results = load_results()

    # Load all datasets
    datasets = {}
    for ds_name in DATASETS:
        datasets[ds_name] = load_dataset(ds_name)

    # Run ETM + TTL sweep on all datasets
    for ds_name in DATASETS:
        train_and_sweep('ETM', ds_name, datasets[ds_name], results)

    # Run ECRTM + TTL sweep on all datasets
    for ds_name in DATASETS:
        train_and_sweep('ECRTM', ds_name, datasets[ds_name], results)

    # Find and save best configs
    for model_name in ['ETM', 'ECRTM']:
        for ds_name in DATASETS:
            best_key = find_best_config(results, model_name, ds_name)
            if best_key:
                summary_key = f'ttl_best_{model_name}_{ds_name}'
                results[summary_key] = dict(results[best_key])
                results[summary_key]['best_config_key'] = best_key

    # Save comparison table
    for ds_name in DATASETS:
        comp_key = f'ttl_comparison_{ds_name}'
        comp = {}
        # ETM baseline
        bk = f'ttl_ETM_baseline_{ds_name}'
        if bk in results:
            comp['ETM_baseline'] = results[bk]
        # ETM + TTL best
        best_etm = find_best_config(results, 'ETM', ds_name)
        if best_etm:
            comp['ETM_TTL_best'] = results[best_etm]
        # ECRTM + TTL best
        best_ecrtm = find_best_config(results, 'ECRTM', ds_name)
        if best_ecrtm:
            comp['ECRTM_TTL_best'] = results[best_ecrtm]
        # EBM-LTM (from existing results)
        ebm_key = f'main_EBMLTM_{ds_name}'
        if ebm_key in results:
            comp['EBMLTM'] = results[ebm_key]
        results[comp_key] = comp

    save_results(results)
    print_summary(results)
    print("\nDone. Results saved to results.json")


if __name__ == '__main__':
    main()
