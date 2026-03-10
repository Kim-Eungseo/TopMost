"""
Multi-seed experiment runner for EBM-LTM paper.
Runs ETM, ETM+Langevin, EBMLTM on 20NG and NYT with 3 seeds.
Reports mean ± std for NPMI, TD, NPMI×TD, Acc, F1.
"""

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmost.data.basic_dataset import BasicDataset
from topmost.models.basic.ETM import ETM
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
NUM_TOP_WORDS = 15
COHERENCE_TOPN = 10
SEEDS = [42, 123, 456]
EPOCHS = {'20NG': 200, 'NYT': 200}
MODELS = ['ETM', 'ETM+Langevin', 'EBMLTM']
DATASETS = ['20NG', 'NYT']


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
    print(f"  Loading {name}...")
    ds = BasicDataset(path, batch_size=BATCH_SIZE, read_labels=True, device=DEVICE)
    return ds


def make_model_and_trainer(model_name, dataset, epochs, seed):
    vocab_size = dataset.vocab_size
    we = dataset.pretrained_WE

    if model_name == 'ETM':
        model = ETM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
        trainer = BasicTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                               epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'ETM+Langevin':
        model = EBMLTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we,
                       lv_steps=15, ls_steps=0,
                       warmup_epochs=int(epochs * 0.25),
                       kl_weight=1.0).to(DEVICE)
        trainer = EBMLTMTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                                epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'EBMLTM':
        model = EBMLTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we,
                       lv_steps=15, ls_steps=30,
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


def run_seed(model_name, ds_name, dataset, seed, results):
    key = f'ms_{model_name}_{ds_name}_seed{seed}'
    if key in results and 'NPMI' in results[key]:
        print(f"  SKIP {key}")
        return results[key]

    print(f"\n{'='*55}")
    print(f"  {model_name} | {ds_name} | seed={seed}")
    print(f"{'='*55}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = EPOCHS[ds_name]
    model, trainer = make_model_and_trainer(model_name, dataset, epochs, seed)

    t0 = time.time()
    try:
        trainer.train()
        elapsed = round(time.time() - t0, 1)
        metrics = evaluate(trainer, dataset)
        metrics['time_sec'] = elapsed
        print(f"  DONE: NPMI={metrics['NPMI']:.4f} TD={metrics['TD']:.4f} Acc={metrics['Acc']:.4f} t={elapsed}s")
    except Exception as e:
        import traceback; traceback.print_exc()
        metrics = {'error': str(e)}

    results[key] = metrics
    save_results(results)
    return metrics


def aggregate_seeds(results, model_name, ds_name):
    """Compute mean ± std across seeds."""
    seed_results = []
    for seed in SEEDS:
        key = f'ms_{model_name}_{ds_name}_seed{seed}'
        r = results.get(key, {})
        if r and 'error' not in r and 'NPMI' in r:
            seed_results.append(r)

    if not seed_results:
        return None

    agg = {}
    for metric in ['NPMI', 'TD', 'NPMI_x_TD', 'Acc', 'F1', 'AU']:
        vals = [r[metric] for r in seed_results if metric in r]
        if vals:
            agg[f'{metric}_mean'] = round(float(np.mean(vals)), 4)
            agg[f'{metric}_std'] = round(float(np.std(vals)), 4)
    return agg


def main():
    results = load_results()

    print("\n" + "="*60)
    print("MULTI-SEED EXPERIMENTS")
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODELS}")
    print(f"Datasets: {DATASETS}")
    print("="*60)

    datasets = {}
    for ds_name in DATASETS:
        datasets[ds_name] = load_dataset(ds_name)

    for ds_name in DATASETS:
        for model_name in MODELS:
            for seed in SEEDS:
                run_seed(model_name, ds_name, datasets[ds_name], seed, results)

    # Save aggregate results
    print("\n" + "="*60)
    print("AGGREGATE RESULTS (mean ± std)")
    print("="*60)

    for ds_name in DATASETS:
        print(f"\n--- {ds_name} ---")
        print(f"{'Model':<20} {'NPMI':>14} {'TD':>12} {'Acc':>12} {'AU':>8}")
        print("-" * 70)
        for model_name in MODELS:
            agg = aggregate_seeds(results, model_name, ds_name)
            if agg:
                key = f'ms_agg_{model_name}_{ds_name}'
                results[key] = agg
                print(f"  {model_name:<18} "
                      f"{agg['NPMI_mean']:.4f}±{agg['NPMI_std']:.4f}  "
                      f"{agg['TD_mean']:.4f}±{agg['TD_std']:.4f}  "
                      f"{agg['Acc_mean']:.4f}±{agg['Acc_std']:.4f}  "
                      f"{agg['AU_mean']:.1f}")

    save_results(results)
    print("\nDone. Results saved to results.json")


if __name__ == '__main__':
    main()
