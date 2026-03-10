"""
EBM-LTM Experiment Runner (v3)

Incremental runs - saves results after each model, resumes on restart.
Datasets: 20NG, IMDB, NYT (3 benchmarks)
Models: ProdLDA, ETM, ECRTM, ETM+Langevin, EBMLTM, EBMLTM+AU
"""

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmost.data.basic_dataset import BasicDataset
from topmost.models.basic.ProdLDA import ProdLDA
from topmost.models.basic.ETM import ETM
from topmost.models.basic.ECRTM.ECRTM import ECRTM
from topmost.models.basic.EBMLTM.EBMLTM import EBMLTM
from topmost.trainers.basic.basic_trainer import BasicTrainer
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
SEED = 42
NUM_TOP_WORDS = 15
COHERENCE_TOPN = 10

# Epochs per dataset
EPOCHS = {'20NG': 200, 'IMDB': 100, 'NYT': 200}


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


def make_model_and_trainer(model_name, dataset, lv_steps=15, epochs=200,
                           au_reg_weight=0.0):
    vocab_size = dataset.vocab_size
    we = dataset.pretrained_WE

    if model_name == 'ProdLDA':
        model = ProdLDA(vocab_size, num_topics=NUM_TOPICS).to(DEVICE)
        trainer = BasicTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                               epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'ETM':
        model = ETM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
        trainer = BasicTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                               epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'ECRTM':
        model = ECRTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
        trainer = BasicTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                               epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'EBMLTM':
        model = EBMLTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we,
                       lv_steps=lv_steps, ls_steps=30,
                       warmup_epochs=int(epochs * 0.25),
                       kl_weight=1.0,
                       au_reg_weight=au_reg_weight).to(DEVICE)
        trainer = EBMLTMTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                                epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'ETM+Langevin':
        # Gaussian prior + Langevin refinement (no energy training)
        model = EBMLTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we,
                       lv_steps=lv_steps, ls_steps=0,
                       warmup_epochs=int(epochs * 0.25),
                       kl_weight=1.0,
                       au_reg_weight=au_reg_weight).to(DEVICE)
        trainer = EBMLTMTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                                epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'EBM_prior_only':
        # Energy prior but no posterior Langevin
        model = EBMLTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we,
                       lv_steps=0, ls_steps=30,
                       warmup_epochs=int(epochs * 0.25),
                       kl_weight=1.0).to(DEVICE)
        trainer = EBMLTMTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                                epochs=epochs, batch_size=BATCH_SIZE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, trainer


def evaluate(trainer, dataset):
    top_words = trainer.get_top_words(NUM_TOP_WORDS)
    train_theta, test_theta = trainer.export_theta()

    # NPMI coherence
    try:
        npmi = _coherence(dataset.train_texts, dataset.vocab, top_words,
                          coherence_type='c_npmi', topn=COHERENCE_TOPN)
    except Exception as e:
        print(f"  NPMI failed ({e}), using c_v")
        try:
            npmi = _coherence(dataset.train_texts, dataset.vocab, top_words,
                              coherence_type='c_v', topn=COHERENCE_TOPN)
        except Exception as e2:
            print(f"  c_v also failed: {e2}")
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


def run_one(model_name, dataset_name, dataset, results, results_key,
            lv_steps=15, au_reg_weight=0.0):
    if results_key in results and 'NPMI' in results[results_key]:
        print(f"  SKIP {results_key} (already done: {results[results_key]})")
        return results[results_key]

    print(f"\n{'='*60}")
    print(f"  {results_key}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    epochs = EPOCHS.get(dataset_name, 200)
    model, trainer = make_model_and_trainer(
        model_name, dataset, lv_steps=lv_steps,
        epochs=epochs, au_reg_weight=au_reg_weight)

    t0 = time.time()
    try:
        trainer.train()
        elapsed = round(time.time() - t0, 1)
        metrics = evaluate(trainer, dataset)
        metrics['time_sec'] = elapsed
        print(f"  DONE: NPMI={metrics['NPMI']:.4f} TD={metrics['TD']:.4f} "
              f"Acc={metrics['Acc']:.4f} AU={metrics['AU']} t={elapsed}s")
    except Exception as e:
        import traceback; traceback.print_exc()
        metrics = {'error': str(e)}
        print(f"  ERROR: {e}")

    results[results_key] = metrics
    save_results(results)
    return metrics


def main():
    results = load_results()

    # Pre-seed ETM+Langevin on 20NG from factorial results (already computed)
    if 'main_ETM+Langevin_20NG' not in results and 'factorial_ETM+Langevin_20NG' in results:
        results['main_ETM+Langevin_20NG'] = results['factorial_ETM+Langevin_20NG']
        save_results(results)
        print("  Pre-seeded main_ETM+Langevin_20NG from factorial result")

    # =====================================================
    # Experiment 1: Main comparison (20NG + IMDB + NYT)
    # =====================================================
    print("\n" + "="*60)
    print("EXP 1: Main Comparison")
    print("="*60)

    datasets_for_main = ['20NG', 'IMDB', 'NYT']
    baselines = ['ProdLDA', 'ETM', 'ECRTM', 'ETM+Langevin', 'EBMLTM']

    datasets = {}
    for ds_name in datasets_for_main:
        datasets[ds_name] = load_dataset(ds_name)

    for ds_name in datasets_for_main:
        dataset = datasets[ds_name]
        for model_name in baselines:
            key = f'main_{model_name}_{ds_name}'
            run_one(model_name, ds_name, dataset, results, key)

    # =====================================================
    # Experiment 2: Langevin steps ablation (20NG only)
    # =====================================================
    print("\n" + "="*60)
    print("EXP 2: Langevin Steps Ablation (20NG)")
    print("="*60)

    dataset_20ng = datasets['20NG']
    for lv in [0, 1, 5, 10, 15, 20]:
        key = f'ablation_lv_{lv}_20NG'
        run_one('EBMLTM', '20NG', dataset_20ng, results, key, lv_steps=lv)

    # =====================================================
    # Experiment 3: 2x2 Factorial ablation (20NG only)
    # =====================================================
    print("\n" + "="*60)
    print("EXP 3: Factorial Ablation (20NG)")
    print("="*60)

    factorial_models = ['ETM', 'ETM+Langevin', 'EBM_prior_only', 'EBMLTM']
    for model_name in factorial_models:
        key = f'factorial_{model_name}_20NG'
        run_one(model_name, '20NG', dataset_20ng, results, key)

    # =====================================================
    # Experiment 4: AU regularizer ablation (20NG)
    # =====================================================
    print("\n" + "="*60)
    print("EXP 4: Active-Unit Regularizer Ablation (20NG)")
    print("="*60)

    for au_w in [0.0, 0.1, 0.5, 1.0]:
        for model_name in ['ETM+Langevin', 'EBMLTM']:
            key = f'au_reg_{model_name}_w{au_w}_20NG'
            # Skip w=0 for ETM+Langevin (same as main result)
            if au_w == 0.0 and model_name == 'ETM+Langevin':
                if key not in results:
                    results[key] = results.get('main_ETM+Langevin_20NG', {})
                    save_results(results)
                continue
            if au_w == 0.0 and model_name == 'EBMLTM':
                if key not in results:
                    results[key] = results.get('main_EBMLTM_20NG', {})
                    save_results(results)
                continue
            run_one(model_name, '20NG', dataset_20ng, results, key,
                    au_reg_weight=au_w)

    # =====================================================
    # Print final summary
    # =====================================================
    print_summary(results)


def print_summary(results):
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    # Main comparison
    print("\n--- TABLE 1: Main Performance Comparison ---")
    hdr = f"{'Model':<18} {'Dataset':<8} {'NPMI':>8} {'TD':>7} {'NPMI×TD':>9} {'Acc':>7} {'F1':>7} {'AU':>4} {'t(s)':>7}"
    print(hdr)
    print("-" * len(hdr))

    for ds in ['20NG', 'IMDB', 'NYT']:
        for m in ['ProdLDA', 'ETM', 'ECRTM', 'ETM+Langevin', 'EBMLTM']:
            k = f'main_{m}_{ds}'
            r = results.get(k, {})
            if 'error' in r:
                print(f"  {m:<16} {ds:<8} ERROR")
                continue
            if not r:
                print(f"  {m:<16} {ds:<8} MISSING")
                continue
            print(f"  {m:<16} {ds:<8} "
                  f"{r.get('NPMI', float('nan')):>8.4f} "
                  f"{r.get('TD', float('nan')):>7.4f} "
                  f"{r.get('NPMI_x_TD', float('nan')):>9.4f} "
                  f"{r.get('Acc', float('nan')):>7.4f} "
                  f"{r.get('F1', float('nan')):>7.4f} "
                  f"{r.get('AU', 0):>4d} "
                  f"{r.get('time_sec', 0):>7.0f}")

    # Langevin ablation
    print("\n--- TABLE 2: Langevin Steps Ablation (20NG) ---")
    print(f"{'lv_steps':<12} {'NPMI':>8} {'TD':>7} {'Acc':>7} {'AU':>4}")
    print("-" * 40)
    for lv in [0, 1, 5, 10, 15, 20]:
        k = f'ablation_lv_{lv}_20NG'
        r = results.get(k, {})
        if 'error' in r or not r:
            continue
        print(f"  {lv:<10} "
              f"{r.get('NPMI', float('nan')):>8.4f} "
              f"{r.get('TD', float('nan')):>7.4f} "
              f"{r.get('Acc', float('nan')):>7.4f} "
              f"{r.get('AU', 0):>4d}")

    # Factorial ablation
    print("\n--- TABLE 3: 2×2 Factorial Ablation (20NG) ---")
    prior_map = {'ETM': 'Gaussian', 'ETM+Langevin': 'Gaussian',
                 'EBM_prior_only': 'EBM', 'EBMLTM': 'EBM'}
    inf_map = {'ETM': 'Amortized', 'ETM+Langevin': '+Langevin',
               'EBM_prior_only': 'Amortized', 'EBMLTM': '+Langevin'}
    print(f"{'Model':<22} {'Prior':<12} {'Inference':<12} {'NPMI':>8} {'TD':>7} {'Acc':>7} {'AU':>4}")
    print("-" * 70)
    for m in ['ETM', 'ETM+Langevin', 'EBM_prior_only', 'EBMLTM']:
        k = f'factorial_{m}_20NG'
        r = results.get(k, {})
        if 'error' in r or not r:
            continue
        print(f"  {m:<20} {prior_map[m]:<12} {inf_map[m]:<12} "
              f"{r.get('NPMI', float('nan')):>8.4f} "
              f"{r.get('TD', float('nan')):>7.4f} "
              f"{r.get('Acc', float('nan')):>7.4f} "
              f"{r.get('AU', 0):>4d}")

    # AU regularizer ablation
    print("\n--- TABLE 4: AU Regularizer Ablation (20NG) ---")
    print(f"{'Model':<20} {'au_w':>6} {'NPMI':>8} {'TD':>7} {'Acc':>7} {'AU':>4}")
    print("-" * 55)
    for au_w in [0.0, 0.1, 0.5, 1.0]:
        for m in ['ETM+Langevin', 'EBMLTM']:
            k = f'au_reg_{m}_w{au_w}_20NG'
            r = results.get(k, {})
            if 'error' in r or not r:
                continue
            print(f"  {m:<18} {au_w:>6.1f} "
                  f"{r.get('NPMI', float('nan')):>8.4f} "
                  f"{r.get('TD', float('nan')):>7.4f} "
                  f"{r.get('Acc', float('nan')):>7.4f} "
                  f"{r.get('AU', 0):>4d}")


if __name__ == '__main__':
    main()
