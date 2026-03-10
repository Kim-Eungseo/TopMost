"""
Extended Baselines: BERTopic and FASTopic.

BERTopic: Cluster-based topic model using TF-IDF (offline, no sentence transformers needed).
FASTopic: Optimal transport-based topic model (if installed).

Evaluates on 20NG and NYT with same NPMI/TD/Acc metrics.
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_ext.json')

NUM_TOPICS = 50
NUM_TOP_WORDS = 15
COHERENCE_TOPN = 10
SEED = 42
DATASETS = ['20NG', 'NYT', 'IMDB']


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_raw_data(ds_name):
    """Load raw text and labels from dataset directory."""
    from topmost.data.basic_dataset import BasicDataset
    import torch
    path = os.path.join(DATA_DIR, ds_name)
    ds = BasicDataset(path, batch_size=200, read_labels=True, device='cpu')
    return ds


def normalize_top_words(top_words):
    """Ensure top_words is a list of space-joined strings."""
    result = []
    for topic in top_words:
        if isinstance(topic, str):
            result.append(topic)
        elif isinstance(topic, (list, tuple)):
            words = []
            for item in topic:
                if isinstance(item, str):
                    words.append(item)
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    words.append(str(item[0]))  # (word, score) tuple
            result.append(' '.join(words))
        else:
            result.append(str(topic))
    return result


def compute_metrics(top_words, train_theta, test_theta, train_texts, vocab, train_labels, test_labels):
    from topmost.eva.topic_coherence import _coherence
    from topmost.eva.topic_diversity import _diversity
    from topmost.eva.classification import _cls
    top_words = normalize_top_words(top_words)

    try:
        # Subsample to 5000 docs for coherence on large corpora (speed)
        coh_texts = train_texts if len(train_texts) <= 5000 else \
            [train_texts[i] for i in np.random.choice(len(train_texts), 5000, replace=False)]
        npmi = _coherence(coh_texts, vocab, top_words,
                          coherence_type='c_npmi', topn=COHERENCE_TOPN)
    except Exception as e:
        print(f"    NPMI failed: {e}")
        npmi = float('nan')

    td = _diversity(top_words)

    try:
        cls_r = _cls(train_theta, test_theta, train_labels, test_labels)
        acc, f1 = cls_r['acc'], cls_r['macro-F1']
    except Exception as e:
        print(f"    Cls failed: {e}")
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


def run_bertopic(ds_name, ds, results):
    """Run BERTopic with TF-IDF (offline mode, no sentence transformers)."""
    key = f'main_BERTopic_{ds_name}'
    if key in results and 'NPMI' in results[key]:
        print(f"  SKIP {key}")
        return results[key]

    print(f"\n{'='*55}")
    print(f"  BERTopic | {ds_name}")
    print(f"{'='*55}")

    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline

        # Use TF-IDF + SVD as embedding (offline, no BERT needed)
        # This is "BERTopic with TF-IDF embeddings" — same cluster approach
        from sklearn.feature_extraction.text import TfidfVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN

        np.random.seed(SEED)

        # Get raw texts from dataset
        train_texts = [' '.join(doc) if isinstance(doc, list) else doc
                       for doc in ds.train_texts]
        test_texts = [' '.join(doc) if isinstance(doc, list) else doc
                      for doc in ds.test_texts]

        # TF-IDF vectorizer for documents -> embeddings via SVD
        print("  Building TF-IDF + SVD embeddings...")
        tfidf = TfidfVectorizer(max_features=10000, min_df=2)
        all_texts = train_texts + test_texts
        tfidf_matrix = tfidf.fit_transform(all_texts)
        svd = TruncatedSVD(n_components=50, random_state=SEED)  # 50 dims for speed
        embeddings_all = svd.fit_transform(tfidf_matrix)
        train_emb = embeddings_all[:len(train_texts)]
        test_emb = embeddings_all[len(train_texts):]

        # Subsample for large corpora (UMAP/HDBSCAN bottleneck)
        MAX_FIT = 8000
        if len(train_texts) > MAX_FIT:
            fit_idx = np.random.choice(len(train_texts), MAX_FIT, replace=False)
            fit_texts = [train_texts[i] for i in fit_idx]
            fit_emb = train_emb[fit_idx]
        else:
            fit_texts, fit_emb = train_texts, train_emb

        print(f"  Embeddings shape: {train_emb.shape}, fitting on: {fit_emb.shape[0]}")

        # UMAP for dimensionality reduction (n_epochs reduced for speed on large corpora)
        umap_model = UMAP(n_neighbors=10, n_components=5, metric='cosine',
                          random_state=SEED, low_memory=True,
                          n_epochs=100, verbose=False)
        # HDBSCAN clustering
        hdbscan_model = HDBSCAN(min_cluster_size=max(10, len(train_texts) // 200),
                                metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True, core_dist_n_jobs=-1)
        # CountVectorizer for topic words
        vectorizer_model = CountVectorizer(vocabulary=ds.vocab,
                                           min_df=1, ngram_range=(1, 1))

        # Build BERTopic
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            nr_topics=NUM_TOPICS,
            top_n_words=NUM_TOP_WORDS,
            calculate_probabilities=True,
            verbose=False,
        )

        t0 = time.time()
        print(f"  Fitting BERTopic on {fit_emb.shape[0]} docs...")
        topics, probs = topic_model.fit_transform(fit_texts, embeddings=fit_emb)
        elapsed_train = time.time() - t0

        # For full train set theta: transform remaining docs
        if len(train_texts) > MAX_FIT:
            all_idx = np.arange(len(train_texts))
            remain_idx = np.setdiff1d(all_idx, fit_idx)
            remain_texts = [train_texts[i] for i in remain_idx]
            remain_emb = train_emb[remain_idx]
            remain_topics, remain_probs = topic_model.transform(remain_texts, embeddings=remain_emb)
            # Reassemble full-dataset topics/probs
            full_topics = np.empty(len(train_texts), dtype=int)
            full_topics[fit_idx] = topics
            full_topics[remain_idx] = remain_topics
            topics = full_topics
            if probs is not None and np.array(probs).ndim == 2:
                full_probs = np.zeros((len(train_texts), np.array(probs).shape[1]))
                full_probs[fit_idx] = np.array(probs)
                if remain_probs is not None and np.array(remain_probs).ndim == 2:
                    full_probs[remain_idx] = np.array(remain_probs)
                probs = full_probs
            else:
                probs = None

        # Get top words per topic
        topic_info = topic_model.get_topics()
        # Filter out topic -1 (outliers) and get top NUM_TOPICS topics
        valid_topics = sorted([k for k in topic_info.keys() if k != -1])[:NUM_TOPICS]
        top_words = []
        for t_id in valid_topics:
            words = [w for w, _ in topic_model.get_topic(t_id)[:NUM_TOP_WORDS]]
            # Filter to vocab
            words = [w for w in words if w in set(ds.vocab)][:NUM_TOP_WORDS]
            if words:
                top_words.append(' '.join(words))  # space-joined string

        # Pad to NUM_TOPICS if needed
        while len(top_words) < NUM_TOPICS:
            top_words.append(' '.join(ds.vocab[:NUM_TOP_WORDS]))

        print(f"  Got {len(top_words)} topics (target: {NUM_TOPICS})")

        # Get document-topic distributions for train
        if probs is not None and len(probs.shape) == 2:
            train_theta = np.array(probs)
        else:
            # One-hot from topic assignments
            train_theta = np.zeros((len(train_texts), NUM_TOPICS))
            for i, t in enumerate(topics):
                if t != -1 and t < NUM_TOPICS:
                    train_theta[i, t] = 1.0

        # Pad/trim columns to NUM_TOPICS
        if train_theta.shape[1] < NUM_TOPICS:
            pad = np.zeros((train_theta.shape[0], NUM_TOPICS - train_theta.shape[1]))
            train_theta = np.concatenate([train_theta, pad], axis=1)
        elif train_theta.shape[1] > NUM_TOPICS:
            train_theta = train_theta[:, :NUM_TOPICS]

        # Transform test set
        print("  Transforming test set...")
        test_topics, test_probs = topic_model.transform(test_texts, embeddings=test_emb)
        if test_probs is not None and len(test_probs.shape) == 2:
            test_theta = np.array(test_probs)
        else:
            test_theta = np.zeros((len(test_texts), NUM_TOPICS))
            for i, t in enumerate(test_topics):
                if t != -1 and t < NUM_TOPICS:
                    test_theta[i, t] = 1.0

        if test_theta.shape[1] < NUM_TOPICS:
            pad = np.zeros((test_theta.shape[0], NUM_TOPICS - test_theta.shape[1]))
            test_theta = np.concatenate([test_theta, pad], axis=1)
        elif test_theta.shape[1] > NUM_TOPICS:
            test_theta = test_theta[:, :NUM_TOPICS]

        elapsed = round(time.time() - t0, 1)

        metrics = compute_metrics(
            top_words, train_theta, test_theta,
            ds.train_texts, ds.vocab,
            ds.train_labels, ds.test_labels
        )
        metrics['time_sec'] = elapsed

        print(f"  DONE: NPMI={metrics['NPMI']:.4f} TD={metrics['TD']:.4f} Acc={metrics['Acc']:.4f} t={elapsed}s")

    except Exception as e:
        import traceback; traceback.print_exc()
        metrics = {'error': str(e)}
        print(f"  ERROR: {e}")

    results[key] = metrics
    save_results(results)
    return metrics


def run_fastopic(ds_name, ds, results):
    """Run FASTopic if installed."""
    key = f'main_FASTopic_{ds_name}'
    if key in results and 'NPMI' in results[key]:
        print(f"  SKIP {key}")
        return results[key]

    print(f"\n{'='*55}")
    print(f"  FASTopic | {ds_name}")
    print(f"{'='*55}")

    try:
        import fastopic
        from fastopic import FASTopic as FASTopicModel
        import torch
        import scipy.sparse

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # FASTopic works with raw text documents
        train_texts = [' '.join(doc) if isinstance(doc, list) else doc
                       for doc in ds.train_texts]
        test_texts = [' '.join(doc) if isinstance(doc, list) else doc
                      for doc in ds.test_texts]

        t0 = time.time()
        model = FASTopicModel(num_topics=NUM_TOPICS)
        model.fit(train_texts)
        elapsed = round(time.time() - t0, 1)

        # Get top words — ensure space-joined string format
        top_words = model.get_top_words(num_top_words=NUM_TOP_WORDS)
        if isinstance(top_words[0], (list, tuple)):
            top_words = [' '.join(t) for t in top_words]
        elif not isinstance(top_words[0], str):
            top_words = [str(t) for t in top_words]

        # Get document-topic distributions
        train_theta = model.transform(train_texts)
        test_theta = model.transform(test_texts)

        if not isinstance(train_theta, np.ndarray):
            train_theta = np.array(train_theta)
        if not isinstance(test_theta, np.ndarray):
            test_theta = np.array(test_theta)

        metrics = compute_metrics(
            top_words, train_theta, test_theta,
            ds.train_texts, ds.vocab,
            ds.train_labels, ds.test_labels
        )
        metrics['time_sec'] = elapsed
        print(f"  DONE: NPMI={metrics['NPMI']:.4f} TD={metrics['TD']:.4f} Acc={metrics['Acc']:.4f} t={elapsed}s")

    except ImportError:
        print("  FASTopic not installed, skipping.")
        metrics = {'error': 'FASTopic not installed'}
    except Exception as e:
        import traceback; traceback.print_exc()
        metrics = {'error': str(e)}
        print(f"  ERROR: {e}")

    results[key] = metrics
    save_results(results)
    return metrics


def print_summary(results):
    print("\n" + "="*70)
    print("EXTENDED BASELINES SUMMARY")
    print("="*70)
    hdr = f"{'Model':<18} {'Dataset':<8} {'NPMI':>8} {'TD':>7} {'Acc':>7} {'AU':>4}"
    print(hdr)
    print("-" * len(hdr))
    for ds in DATASETS:
        for m in ['ETM', 'ECRTM', 'ETM+Langevin', 'EBMLTM', 'BERTopic', 'FASTopic']:
            k = f'main_{m}_{ds}'
            r = results.get(k, {})
            if 'error' in r or not r:
                continue
            print(f"  {m:<16} {ds:<8} "
                  f"{r.get('NPMI', float('nan')):>8.4f} "
                  f"{r.get('TD', float('nan')):>7.4f} "
                  f"{r.get('Acc', float('nan')):>7.4f} "
                  f"{r.get('AU', 0):>4d}")


def main():
    results = load_results()

    for ds_name in DATASETS:
        print(f"\n\nDataset: {ds_name}")
        ds = load_raw_data(ds_name)
        run_bertopic(ds_name, ds, results)
        run_fastopic(ds_name, ds, results)

    print_summary(results)
    print("\nDone.")


if __name__ == '__main__':
    main()
