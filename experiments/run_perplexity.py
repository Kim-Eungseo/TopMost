"""
Held-Out Perplexity Experiment for EBM-LTM Paper.

Computes PPL-amortized and PPL-refined for:
  ETM, ECRTM, ETM+Langevin, EBM-LTM
on 20NG, NYT, IMDB.

BERTopic / FASTopic: reported as --- (no generative likelihood).
Saves to experiments/results.json with keys: ppl_{model}_{dataset}
"""

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmost.data.basic_dataset import BasicDataset
from topmost.models.basic.ETM import ETM
from topmost.models.basic.ECRTM.ECRTM import ECRTM
from topmost.models.basic.EBMLTM.EBMLTM import EBMLTM
from topmost.trainers.basic.basic_trainer import BasicTrainer
from topmost.trainers.basic.EBMLTM_trainer import EBMLTMTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

NUM_TOPICS = 50
BATCH_SIZE = 200
NUM_TOP_WORDS = 15
SEED = 42
EPOCHS = {'20NG': 200, 'IMDB': 100, 'NYT': 200}
DATASETS = ['20NG', 'NYT', 'IMDB']
MODELS = ['ETM', 'ECRTM', 'ETM+Langevin', 'EBMLTM']


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


def make_model_trainer(model_name, dataset, epochs):
    vocab_size = dataset.vocab_size
    we = dataset.pretrained_WE

    if model_name == 'ETM':
        model = ETM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
        trainer = BasicTrainer(model, dataset, num_top_words=NUM_TOP_WORDS,
                               epochs=epochs, batch_size=BATCH_SIZE)

    elif model_name == 'ECRTM':
        model = ECRTM(vocab_size, num_topics=NUM_TOPICS, pretrained_WE=we).to(DEVICE)
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

    return model, trainer


def run_ppl(model_name, ds_name, dataset, results):
    key = f'ppl_{model_name}_{ds_name}'
    if key in results and 'ppl_amortized' in results[key]:
        print(f"  SKIP {key}")
        return results[key]

    print(f"\n{'='*55}")
    print(f"  PPL | {model_name} | {ds_name}")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    epochs = EPOCHS[ds_name]
    model, trainer = make_model_trainer(model_name, dataset, epochs)

    t0 = time.time()
    try:
        trainer.train()
        elapsed_train = round(time.time() - t0, 1)

        t1 = time.time()
        ppl_dict = trainer.held_out_perplexity()
        elapsed_ppl = round(time.time() - t1, 1)

        r = {
            'ppl_amortized': ppl_dict['ppl_amortized'],
            'ppl_refined': ppl_dict['ppl_refined'],
            'train_time_sec': elapsed_train,
            'ppl_time_sec': elapsed_ppl,
        }
        pa = r['ppl_amortized']
        pr = r['ppl_refined']
        print(f"  DONE: PPL-amort={pa:.2f}  PPL-refined={pr}  train={elapsed_train}s")

    except Exception as e:
        import traceback; traceback.print_exc()
        r = {'error': str(e)}
        print(f"  ERROR: {e}")

    results[key] = r
    save_results(results)
    return r


def print_summary(results):
    print("\n" + "="*70)
    print("HELD-OUT PERPLEXITY SUMMARY")
    print("="*70)
    hdr = f"{'Model':<18} {'Dataset':<8} {'PPL-amort':>12} {'PPL-refined':>12}"
    print(hdr)
    print("-" * len(hdr))
    for ds in DATASETS:
        for m in MODELS:
            k = f'ppl_{m}_{ds}'
            r = results.get(k, {})
            if 'error' in r or not r:
                continue
            pa = r.get('ppl_amortized', float('nan'))
            pr = r.get('ppl_refined', None)
            pr_str = f"{pr:.2f}" if pr is not None else "  ---"
            print(f"  {m:<16} {ds:<8} {pa:>12.2f} {pr_str:>12}")

    # BERTopic / FASTopic
    for ds in DATASETS:
        for m in ['BERTopic', 'FASTopic']:
            print(f"  {m:<16} {ds:<8} {'---':>12} {'---':>12}")


def main():
    results = load_results()

    for ds_name in DATASETS:
        print(f"\n\nDataset: {ds_name}")
        dataset = load_dataset(ds_name)
        for model_name in MODELS:
            run_ppl(model_name, ds_name, dataset, results)

    print_summary(results)
    print("\nDone. Results saved to results.json")


if __name__ == '__main__':
    main()
